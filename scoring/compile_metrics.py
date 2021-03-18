#!/usr/bin/env python
#
# Parses and compiles metrics previously computed by calc_metrics.sh.
#
# Usage: ./compile_metrics.py FOLDER
# 
# FOLDER should contain subfolders like lddt, trr_score, etc. This will output
# a file, combined_metrics.csv, in FOLDER.
#
import pandas as pd
import numpy as np
import os, glob, argparse

p = argparse.ArgumentParser()
p.add_argument('folder', help='Folder of outputs to process')
p.add_argument('--out',  help='Output file name.')
args = p.parse_args()

if args.out is None:
    args.out = os.path.join(args.folder,'combined_metrics.csv')

def parse_fastdesign_filters(folder):
    files = glob.glob(os.path.join(folder,'*.pdb'))
    df = pd.DataFrame()
    for f in files:
        row = pd.DataFrame()
        row['name'] = [os.path.basename(f)[:-4]]
        recording = False
        with open(f) as inf:
            for line in inf:
                if recording and len(line)>1:
                    k,v = line.split()
                    row[k] = float(v)
                if '#END_POSE_ENERGIES_TABLE' in line: 
                    recording=True
                if line.startswith('pose'):
                    row['rosetta_energy'] = float(line.split()[-1])
        df = df.append(row)
    return df

def parse_lddt(folder):
    data = {'name':[], 'lddt':[]}
    files = glob.glob(os.path.join(folder,'*.npz'))
    if len(files)==0:
        return pd.DataFrame()
    for f in files:
        prefix = os.path.basename(f).replace('.npz','')
        lddt_data = np.load(f)
        data['lddt'].append(lddt_data['lddt'].mean())
        data['name'].append(prefix)
    return pd.DataFrame.from_dict(data)

def parse_rosetta_energy_from_pdb(folder):
    files = glob.glob(os.path.join(folder,'*.pdb'))
    if len(files)==0:
        return pd.DataFrame()
    df = pd.DataFrame()
    for pdbfile in files:
        with open(pdbfile) as inf:
            name = os.path.basename(pdbfile).replace('.pdb','')
            rosetta_energy = np.nan
            for line in inf.readlines():
                if line.startswith('pose'):
                    rosetta_energy = float(line.split()[-1])
            row = pd.DataFrame()
            row['name'] = [name]
            row['rosetta_energy'] = rosetta_energy
            df = df.append(row)
    return df

def parse_frag_qual(folder):
    df = pd.DataFrame()
    for frag_folder in glob.glob(os.path.join(folder,'*_fragments')):
        fn = os.path.join(frag_folder,'frag_qual.dat')
        if not os.path.exists(fn): continue
        with open(fn) as inf:
            lines = inf.readlines()
            index=1
            y_index=[]
            y_avg=[]
            y_bestmer=[]
            for line in lines:
                if int(line.split()[1]) == index:
                    y_index.append(float(line.split()[3]))
                else:
                    y_avg.append(np.average(np.array(y_index)))
                    y_bestmer.append(np.amin(np.array(y_index)))
                    y_index=[]
                index=int(line.split()[1])
            avg_all_frags=np.average(y_avg)
            avg_best_frags=np.average(y_bestmer)
        row = pd.DataFrame()
        row['name'] = [os.path.basename(frag_folder).replace('_fragments','')]
        row['avg_all_frags'] = avg_all_frags
        row['avg_best_frags'] = avg_best_frags
        df = df.append(row)
    return df

def parse_cce(folder):
    df = pd.DataFrame()
    for fn in glob.glob(os.path.join(folder,'*.trR_scored.txt')):
        row = pd.read_csv(fn)
        df = df.append(row)
    if df.shape[0]>0:
        df.columns = ['name','cce10','cce_1d','acc']
        return df[['name','cce10']]
    else:
        return df

def parse_all_metrics(folder):
    df = pd.DataFrame()
    df['name'] = []

    print(f'Parsing metrics in {folder}: ',end='')
    tmp = parse_lddt(os.path.join(folder,'lddt'))
    if tmp.shape[0]>0:
        df = df.merge(tmp,on='name',how='outer')
        print(f'lddt ({tmp.shape[0]}), ',end='')

    fn = os.path.join(folder,'pymol_metrics.csv')
    if os.path.exists(fn):
        tmp = pd.read_csv(fn,index_col=0)
        df = df.merge(tmp,on='name',how='outer')
        print(f'pymol metrics ({tmp.shape[0]}), ',end='')

    tmp = parse_fastdesign_filters(os.path.join(folder))
    if tmp.shape[0]>0:
        df = df.merge(tmp,on='name',how='outer')
        print(f'fastdesign metrics ({tmp.shape[0]}), ',end='')

    tmp = parse_cce(os.path.join(folder,'trr_score'))
    if tmp.shape[0]>0:
        df = df.merge(tmp,on='name',how='outer')
        print(f'cce ({tmp.shape[0]}), ',end='')

    tmp = parse_frag_qual(os.path.join(folder,'frags'))
    if tmp.shape[0]>0:
        df = df.merge(tmp,on='name',how='outer')
        print(f'fragment quality ({tmp.shape[0]}), ',end='')

    fn = os.path.join(folder,'ss_frac.csv')
    if os.path.exists(fn):
        tmp = pd.read_csv(fn,index_col=0)
        df = df.merge(tmp,on='name',how='outer')
        print(f'sec. struct. frac ({tmp.shape[0]}), ',end='')

    print(f'final dataframe shape: {df.shape}')
    return df

df = parse_all_metrics(args.folder)
df.to_csv(args.out)
