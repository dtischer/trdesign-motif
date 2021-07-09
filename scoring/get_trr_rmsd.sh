#!/bin/bash
#
# Computes rmsd and other metrics from structural predictions from PyTorch
# implementation of TrRosetta
#
# Usage: ./get_trr_rmsd.sh FOLDER_OF_PDBS
#
# 2021-4-10

sbatch -p gpu --gres gpu:rtx2080:1 -c 2 --mem 16g -J rmsd_trr.`basename $1` --wrap="/home/jue/git/bff/design/score.py --network=trr2_msa_v00_l --pdb-dir $1 --ocsv $1/rmsd_trr.csv"
