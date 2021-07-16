#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 2
#SBATCH --job-name=lddt_bb
source /software/conda/etc/profile.d/conda.sh
conda activate tensorflow

mkdir -p $1/lddt-bb
SCRIPT=/home/hiranumn/Backbone-related/BackboneEstimator-FULL.py/BackboneEstimator-FULL.py
python $SCRIPT $1 $1/lddt-bb
