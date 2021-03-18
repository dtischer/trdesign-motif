#!/software/conda/envs/pyrosetta/bin/python

# Calculates fraction of each sequence that is strand, helix, or loop secondary
# structure, given PDB. Operates on a folder of PDBs and outputs results to
# CSV.
#
# Usage:
#
#     ./get_ss_frac.py PDB_FOLDER
#
# By default the result is saved to PDB_FOLDER/ss_frac.csv. You can also
# specify the output file:
#
#     ./get_ss_frac.py -o some/other/location.csv PDB_FOLDER
#
# Updated 2021-3-17


import pandas as pd
import numpy as np
import argparse, os, glob
import pyrosetta as pyr

p = argparse.ArgumentParser()
p.add_argument('folder', help='Folder of PDBs to process')
p.add_argument('-o','--out', help=('Output filename (csv).'))
args = p.parse_args()

pyr.init()

df = pd.DataFrame()
for f in glob.glob(os.path.join(args.folder,'*.pdb')):
    row = pd.Series()
    pose = pyr.pose_from_pdb(f)
    DSSP = pyr.rosetta.protocols.moves.DsspMover()
    DSSP.apply(pose)
    ss = pose.secstruct()
    
    row['name'] = os.path.basename(f).replace('.pdb','')
    row['len'] = len(pose.sequence())
    row['ss_strand_frac'] = ss.count('E') / row['len']
    row['ss_helix_frac'] = ss.count('H') / row['len']
    row['ss_loop_frac'] = ss.count('L') / row['len']
    
    df = df.append(row,ignore_index=True)
df = df.reset_index(drop=True)

if args.out is not None:
    outfile = args.out
else:
    outfile = os.path.join(args.folder, 'ss_frac.csv')
df.to_csv(outfile)
