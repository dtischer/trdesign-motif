#!/software/conda/envs/pyrosetta/bin/python
# 
# Calculates motif RMSDs between design models and reference structure for a
# folder of PDBs and outputs to a CSV. Also calculates radius of gyration,
# length, secondary structure fraction, and adds losses from trb file to CSV. 
#
# Usage:
#
#     ./pyrosetta_metrics.py DESIGNS_FOLDER
#
# This calculates the RMSD to the template given in the .trb file for each
# design. If you'd like to provide the template, use:
#
#     ./pyrosetta_metrics.py --template TEMPLATE_PDB DESIGNS_FOLDER
#
# Passing a file with space-delimited residue numbers to --interface_res will
# also output an 'interface_rmsd' on only the residues in the file (and in the
# contigs in the .trb file):
#
#     ./pyrosetta_metrics.py --template TEMPLATE_PDB --interface_res RES_FILE
#     DESIGNS_FOLDER
#
# Updated 2021-7-8

import pandas as pd
import numpy as np
import sys, os, argparse, glob
from collections import OrderedDict

import pyrosetta
pyrosetta.init('-mute all')

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
p.add_argument('--sc_rmsd', action='store_true', default=True, 
    help='Also calculate side-chain RMSD, returning NaN if residues aren\'t matched.')
p.add_argument('--interface_res', 
    help='File with space-separated integers of residue positions. Report rmsd on these '\
         'residues as "interface_rmsd"')
p.add_argument('-o','--out', help=('Prefix of output filenames.'))
p.add_argument('--trb_dir', help='Folder containing .trb files (if not same as pdb folder)')
p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')
args = p.parse_args()

def calc_rmsd(xyz_ref, xyz_hal, eps=1e-6):

    # center to CA centroid
    xyz_ref = xyz_ref - xyz_ref.mean(0)
    xyz_hal = xyz_hal - xyz_hal.mean(0)

    # Computation of the covariance matrix
    C = xyz_hal.T @ xyz_ref

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz_hal
    rP = xyz_hal @ U

    L = rP.shape[0]
    rmsd = np.sqrt(np.sum((rP-xyz_ref)*(rP-xyz_ref), axis=(0,1)) / L + eps)

    return rmsd

def get_rmsd(pose_ref, pose_hal, trb, mode='ca', interface_res=None):

    if mode=='ca':
        atoms = ["CA"]

    xyz_ref = []
    xyz_hal = []

    if 'con_ref_pdb_idx' not in trb:
        trb['con_ref_pdb_idx'] = [('A',i) for i in trb['con_ref_idx0'][0]]
        trb['con_hal_pdb_idx'] = [('A',i+1) for i in trb['con_hal_idx0'][0]]

    for idx_ref, idx_hal in zip(trb['con_ref_pdb_idx'], trb['con_hal_pdb_idx']):
        if interface_res is not None and idx_ref not in interface_res:
            continue

        # Find equivalent residues in both structures
        pose_idx_ref = pose_ref.pdb_info().pdb2pose(*idx_ref)
        pose_idx_hal = pose_hal.pdb_info().pdb2pose(*idx_hal)

        res_hal = pose_hal.residue(pose_idx_hal)
        res_ref = pose_ref.residue(pose_idx_ref)

        if mode=='sc':
            if res_hal.name3() != res_ref.name3():
                return np.nan
            atoms_ref = [res_ref.atom_name(i) for i in range(5,res_ref.natoms()+1)]
            atoms_hal = [res_hal.atom_name(i) for i in range(5,res_hal.natoms()+1)]
            atoms = [a for a in atoms_ref if a in atoms_hal]

        for atom in atoms:
            xyz_ref.append(np.array(res_ref.atom(res_ref.atom_index(atom)).xyz()))
            xyz_hal.append(np.array(res_hal.atom(res_hal.atom_index(atom)).xyz()))

    xyz_ref = np.array(xyz_ref)
    xyz_hal = np.array(xyz_hal)

    return calc_rmsd(xyz_ref, xyz_hal)

def main():

    # input and output names
    if args.out is not None:
        outfile = args.out if '.csv' in args.out else args.out+'.csv'
    else:
        outfile = os.path.join(args.data_dir,'pyrosetta_metrics.csv')

    # for radius of gyration
    rog_scorefxn = pyrosetta.ScoreFunction()
    rog_scorefxn.set_weight( pyrosetta.rosetta.core.scoring.ScoreType.rg , 1 )

    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()

    if args.template is not None:
        pose_ref_clean = pyrosetta.pose_from_file(args.template)
    else:
        last_template = ''

    # calculate contig RMSD
    print(f'Calculating RMSDs')

    df = pd.DataFrame()
    if args.trb_dir is not None: trb_dir = args.trb_dir
    else: trb_dir = args.data_dir

    records = []
    for fn in glob.glob(os.path.join(args.data_dir,'*.pdb')):
        row = OrderedDict()
        row['name'] = os.path.basename(fn).replace('.pdb','')
        print(row['name'])

        trbname = os.path.join(trb_dir, os.path.basename(fn.replace(args.pdb_suffix+'.pdb','.trb')))
        if not os.path.exists(trbname): 
            sys.exit(f'ERROR: {trbname} does not exist. Set the --trb_dir argument if your .trb files '\
                      'are in a different folder from the .pdb files.')
        trb = np.load(trbname,allow_pickle=True)

        # save final losses
        for k in trb:
            if k.startswith('loss'):
                if type(trb[k]) is list:
                    row[k] = trb[k][0]
                else:
                    row[k] = trb[k]

        if args.template is None and 'settings' in trb and trb['settings']['pdb'] != last_template:
            pose_ref_clean = pyrosetta.pose_from_file(trb['settings']['pdb'])
            last_template = trb['settings']['pdb']

        pose_ref = pose_ref_clean.clone()
        pose_hal = pyrosetta.pose_from_file(fn)

        row['contig_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='ca')
        print('contig_rmsd: ', row['contig_rmsd'])
        if args.sc_rmsd:
            row['contig_sc_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='sc')
            print('contig_sc_rmsd: ', row['contig_sc_rmsd'])

        if args.interface_res is not None:
            interface_res = []
            for line in open(args.interface_res).readlines():
                for x in line.split():
                    if x[0].isalpha():
                        interface_res.append((x[0], int(x[1:])))
                    else:
                        interface_res.append(('A', int(x)))

            row['interface_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='ca', interface_res=interface_res)
            print('interface_rmsd: ', row['interface_rmsd'])
            if args.sc_rmsd:
                row['interface_sc_rmsd'] = get_rmsd(pose_ref, pose_hal, trb, mode='sc', interface_res=interface_res)
                print('interface_sc_rmsd: ', row['interface_sc_rmsd'])

        row['rog'] = rog_scorefxn( pose_hal )

        DSSP.apply(pose_hal)
        ss = pose_hal.secstruct()

        row['len'] = len(pose_hal.sequence())
        row['ss_strand_frac'] = ss.count('E') / row['len']
        row['ss_helix_frac'] = ss.count('H') / row['len']
        row['ss_loop_frac'] = ss.count('L') / row['len']

        records.append(row)

    df = pd.DataFrame.from_records(records)

    print(f'Outputting computed metrics to {outfile}')
    df.to_csv(outfile)

if __name__ == "__main__":
    main()
