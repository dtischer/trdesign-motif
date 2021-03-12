#!/bin/bash
#
# Usage: ./calc_cce.sh FOLDER_OF_PDBS
#
# Optional: ./calc_cce.sh FOLDER_OF_PDBS JOB_NAME
#

outdir=$1/trr_score
mkdir -p $outdir

if [ "$#" -lt 2 ]; then
    jobname=calc_cce.sh
else
    jobname=$2
fi

sbatch --mem=8g -p gpu --gres=gpu:rtx2080:1 -J $jobname --wrap="/software/conda/envs/tensorflow/bin/python /home/jue/trDesign/TrR_for_design_v4/predict.py --pdb=$1 --folder --save_loss --cce_cutoff=10 --out=$outdir"

