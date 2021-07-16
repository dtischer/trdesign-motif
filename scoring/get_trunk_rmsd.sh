#!/bin/bash
#
# Computes rmsd and other metrics from structural predictions from AF2-like
# Trunk module
#
# Usage: ./get_trunk_rmsd.sh FOLDER_OF_PDBS
#
# 2021-4-10

sbatch -p gpu --gres gpu:rtx2080:1 -c 2 --mem 16g -J rmsd_trunk.`basename $1` --wrap="/home/jue/git/bff/design/score.py --network=trunk_tbm_v00 --pdb-dir $1 --ocsv $1/rmsd_trunk.csv"
