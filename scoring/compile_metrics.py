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
    print(f'Parsing metrics in {folder}: ',end='')
    df_lddt = parse_lddt(os.path.join(folder,'lddt'))
    if df_lddt.shape[0]==0: df_lddt['name']=[]
    print(f'lddt ({df_lddt.shape[0]}), ',end='')

    fn = os.path.join(folder,'pymol_metrics.csv')
    if os.path.exists(fn):
        df_py = pd.read_csv(fn,index_col=0)
    else:
        df_py = pd.DataFrame({'name':[]})
    print(f'pymol metrics ({df_py.shape[0]}), ',end='')

    df_fd = parse_fastdesign_filters(os.path.join(folder))
    if df_fd.shape[0]==0: df_fd['name']=[]
    print(f'fastdesign metrics ({df_fd.shape[0]}), ',end='')

    df_cce = parse_cce(os.path.join(folder,'trr_score'))
    if df_cce.shape[0]==0: df_cce['name']=[]
    print(f'cce ({df_cce.shape[0]}), ',end='')

    df_frag = parse_frag_qual(os.path.join(folder,'frags'))
    if df_frag.shape[0]==0: df_frag['name']=[]
    print(f'fragment quality ({df_frag.shape[0]}), ',end='')

    tmp = (df_lddt.merge(df_py,on='name',how='outer')
           .merge(df_fd, on = 'name',how='outer')
           .merge(df_cce, on = 'name',how='outer')
           .merge(df_frag, on = 'name',how='outer'))
    print(f'final dataframe shape: {tmp.shape}')
    return tmp

df = parse_all_metrics(args.folder)
df.to_csv(args.out)
