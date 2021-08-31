#!/bin/bash
source activate /software/conda/envs/tensorflow

mkdir -p $1/lddt
SCRIPT=/home/jue/git/DeepAccNet/DeepAccNet.py

if [ -z "$2" ]; then
    jobname=lddt
else
    jobname=$2
fi

sbatch -J $jobname -p gpu --mem=8g --gres=gpu:1 -c 2 -o /dev/null -e /dev/null \
    --wrap="python $SCRIPT $1 $1/lddt &>$1/lddt/log.txt"
