#!/bin/bash
#
# usage: sbatch pick_frags_calc_qual.sh <pdb_folder>
#
# outputs will be in <pdb_folder>/frags
#
# Updated 2020-12-30
#
#SBATCH -c 20
#SBATCH --mem=48g

# pick fragments
mkdir -p $1/frags
cd $1/frags
/home/robetta/workspace/labFragPicker_DO_NOT_REMOVE/bakerlab_scripts/boinc/make_fragments.py -pdbs_folder .. -n_proc 20

# calculate fragment quality
for i in *_fragments; do
    cd $i
    echo $i
    if [ ! -f frag_qual.dat ]; then
        /home/dekim/bin/clean_header_pdb.pl 00001.pdb
        /home/robetta/workspace/labFragPicker_DO_NOT_REMOVE/Rosetta/main/source/bin/r_frag_quality.linuxgccrelease -in:file:native 00001.pdb -f 00001.200.9mers
        rm 0000*
    fi
    rm core
    cd ..
done
