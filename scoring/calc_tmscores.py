#!/usr/bin/env python

# Computes TM score between set of PDB structures and a template structure.
#
# Usage: ./get_tmscores.py -t template.pdb -o results.csv PDB1.pdb, PDB2.pdb ...
#
# 2021-3-21

import pandas as pd
import numpy as np
import glob, os, argparse, re, subprocess

p = argparse.ArgumentParser()
p.add_argument('data_dir',help='Folder of PDB files to align to template')
p.add_argument('-t', '--template', required=True, help='Single PDB file to align to.')
p.add_argument('-o','--out', help='Output file (.csv)')
args = p.parse_args()

if args.out is None:
    outfile = os.path.join(args.data_dir,'tmscores.csv')
else:
    outfile = args.out

df = pd.DataFrame()
for fn in glob.glob(os.path.join(args.data_dir,'*.pdb')):
    name = os.path.basename(fn.replace('.pdb',''))
    cmd = f'/home/aivan/prog/TMscore {args.template}'
    if os.path.exists(fn):
        out = subprocess.getoutput(f'{cmd} {fn}')
        m = re.search('TM\-score\s+?\= (\d\.\d+)',out)
        tm = float(m.groups()[0])
        m = re.search('GDT\-TS\-score\= (\d\.\d+)',out)
        gdt = float(m.groups()[0])

        df = df.append(pd.Series({
            'name':name,
            'tm_vs_template':tm,
            #'gdt_vs_template':gdt,
        }), ignore_index=True)

df.to_csv(outfile)

