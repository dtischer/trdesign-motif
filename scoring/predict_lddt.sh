#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o /dev/null # don't generate slurm.txt logs (but do make real log file below)
#SBATCH -e /dev/null
#SBATCH --job-name=lddt_predict
source activate /software/conda/envs/tensorflow

mkdir -p $1/lddt
SCRIPT=/home/jue/git/DeepAccNet/DeepAccNet.py
python $SCRIPT $1 $1/lddt &>$1/lddt/log.txt
