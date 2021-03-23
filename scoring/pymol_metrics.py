#!/usr/bin/pymol -qc

# 
# Calculates motif RMSDs between design models and reference structure for a
# folder of PDBs and outputs to a CSV. Also calculates radius of gyration and
# adds losses from trb file to CSV. If receptor .pdb is supplied, will also
# compute clashes between design and receptor after aligning to template
# binder.
#
# Usage:
#
#     ./pymol_metrics.py --template TEMPLATE_PDB --receptor RECEPTOR_PDB DESIGNS_FOLDER
#
# Passing a file with space-delimited residue numbers to --interface_res will
# also output an 'interface_rmsd' on only the residues in the file (and in the
# contigs in the .trb file):
#
#     ./pymol_metrics.py --template TEMPLATE_PDB --receptor RECEPTOR_PDB \
#                        --interface_res RES_FILE DESIGNS_FOLDER
#
# Help:
#
#     ./pymol_metrics.py -- --help
#
# Updated 2021-3-14

import pandas as pd
import numpy as np
from pymol import cmd,stored
from glob import glob
from os.path import dirname, basename, exists
import argparse

p = argparse.ArgumentParser()
p.add_argument('data_dir', help='Folder of TrDesign outputs to process')
p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
p.add_argument('-r','--receptor', help='Receptor structure (.pdb). Should be in same coordinate ' \
                                                                             'frame as template PDB.')
p.add_argument('--interface_res', help='Only align positions in this file (text file ' \
                                       'with space delimited integers)')
p.add_argument('-o','--out', help=('Prefix of output filenames.'))
p.add_argument('--trb_dir', help='Folder containing .trb files (if not same as pdb folder)')
p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')

p.add_argument('--clashdist', type=int, default=2, help='Distance (angstroms) from receptor to consider a clash.')
p.add_argument('--nclashes', type=int, default=0, help='Allow this many or fewer clashes with receptor.')
args = p.parse_args()

def idx2contigstr(idx):
    istart = 0
    contigs = []
    for iend in np.where(np.diff(idx)!=1)[0]:
            contigs += [f'{idx[istart]}-{idx[iend]}']
            istart = iend+1
    contigs += [f'{idx[istart]}-{idx[-1]}']
    return contigs

def get_contig_idx(trk):
    if 'con_ref_pdb_idx' in trk:
        refkey, halkey = 'con_ref_pdb_idx','con_hal_pdb_idx'
        refchain, refidx = zip(*trk[refkey])
        if np.unique(refchain).size > 1:
            sys.exit('ERROR: This script does not support reference contigs on different chains.')
        halidx = [x[1] for x in trk[halkey]]
    else:
        if 'con_ref_idx0' in trk: refkey, halkey = 'con_ref_idx0','con_hal_idx0'
        else: refkey, halkey = 'con_idxs_ref','con_idxs_hal'
        refidx = trk[refkey][0]
        halidx = trk[halkey][0]+1
    return np.array(refidx,dtype=int), np.array(halidx,dtype=int)

def get_rmsd(refidx, halidx, refname, halname):

    # get residue numbers
    stored.idx = []
    cmd.iterate(refname,'stored.idx.append(resi)')
    ref_idx_all = np.array([int(i) for i in stored.idx])
    stored.idx = []
    cmd.iterate(halname,'stored.idx.append(resi)')
    hal_idx_all = np.array([int(i) for i in stored.idx])

    # remove contig positions not present in halluc (due to trimming)
    mask = np.isin(halidx,hal_idx_all)
    halidx = halidx[mask]
    refidx = refidx[mask]
    
    # renumber contigs so selection residues are monotonically increasing
    contigs = idx2contigstr(refidx)
    istart = [int(x.split('-')[0]) for x in contigs]
    istop = [int(x.split('-')[1]) for x in contigs]

    iunoccupied = max(ref_idx_all)+1
    offsets = []
    imax = 0
    for i1,i2 in zip(istart,istop):
        if i1<imax:
            offset = iunoccupied-i1
            cmd.alter(f'{refname}/{i1}-{i2}',f'resi=int(resi)+{offset}')
            iunoccupied += i2 - i1 + 1 # range includes i2
        else:
            offset = 0
        offsets.append(offset)
        if i2>imax: imax = i2
    cmd.sort()

    cstr1 = "+".join([f'{i1+off}-{i2+off}' for i1,i2,off in zip(istart,istop,offsets)])
    cstr2 = "+".join(idx2contigstr(halidx))
    #print(f'{ref_prefix}/{cstr1}/CA, {row["name"]}///{cstr2}/CA')
    rms = cmd.pair_fit(f'{refname}/{cstr1}/CA',f'{halname}///{cstr2}/CA')
    for i1,i2,off in zip(istart,istop,offsets):
        cmd.alter(f'{refname}/{i1+off}-{i2+off}',f'resi=int(resi)-{off}')
    cmd.sort()
    #stored.idx = []
    #cmd.iterate(ref_prefix+'//CA','print(resi)')

    # detect clashes
    receptor = os.path.basename(args.receptor).replace('.pdb','')
    n = cmd.select(f'{halname}////CA and all within {args.clashdist} of {receptor}')

    return rms, n

def calc_rog(selection):
    model = cmd.get_model(selection).atom
    x = [i.coord for i in model]
    mass = [i.get_mass() for i in model]
    xm = [(m*i,m*j,m*k) for (i,j,k),m in zip(x,mass)]
    tmass = sum(mass)
    rr = sum(mi*i+mj*j+mk*k for (i,j,k),(mi,mj,mk) in zip(x,xm))
    mm = sum((sum(i)/tmass)**2 for i in zip(*xm))
    rg = math.sqrt(rr/tmass - mm)
    return rg

# input and output names
ref_prefix = f'/{os.path.basename(args.template).replace(".pdb","")}//'
tokens = args.data_dir.split('/')
folder = tokens[-1] if tokens[-1] else tokens[-2]
if args.out is not None:
    outfile = args.out if '.csv' in args.out else args.out+'.csv'
else:
    outfile = os.path.join(args.data_dir,'pymol_metrics.csv')

# calculate contig RMSD
print(f'Calculating RMSDs and receptor clashes')

df = pd.DataFrame()
if args.trb_dir is not None: trb_dir = args.trb_dir
else: trb_dir = args.data_dir

for f in glob(os.path.join(args.data_dir,'*.pdb')):
    row = pd.Series()
    row['name'] = os.path.basename(f).replace('.pdb','')
    print(row['name'])

    trbname = os.path.join(trb_dir, os.path.basename(f.replace(args.pdb_suffix+'.pdb','.trb')))
    if not exists(trbname): 
        print('{trbname} does not exist, skipping')
        continue
    trk = np.load(trbname,allow_pickle=True)

    if 'loss_nodrop' in trk:
        loss = trk['loss_nodrop']
        for k in loss:
            row[k] = loss[k]
        row['total_loss'] = sum([loss[k] for k in loss])

    cmd.reinitialize()
    cmd.load(args.template)
    if args.receptor is not None:
        cmd.load(args.receptor)
    cmd.load(f)

    refidx, halidx = get_contig_idx(trk)
    if np.unique(refidx).shape!=refidx.shape: continue # skip designs with contig bug

    rms, nclashes = get_rmsd(refidx, halidx, ref_prefix, row['name'])
    row['contig_rmsd'] = rms
    row['rec_clashes_contig'] = nclashes

    if args.interface_res is not None:
        idx = [int(x) for line in open(args.interface_res).readlines()
                      for x in line.split()]
        mask = np.isin(refidx, idx)
        refidx = refidx[mask]
        halidx = halidx[mask]
        rms, nclashes = get_rmsd(refidx, halidx, ref_prefix, row['name'])
        row['interface_rmsd'] = rms
        row['rec_clashes_interface'] = nclashes

    row['rog'] = calc_rog(row['name'])

    cmd.remove(row['name'])
    cmd.delete(row['name'])

    df = df.append(row, ignore_index=True)

df = df.reset_index(drop=True)

print(f'Outputting computed metrics to {outfile}')
df.to_csv(outfile)

