#!/bin/bash
#
# Calculate lddt, cce10, avg_all_frags: 
#
#   ./calc_metrics.sh FOLDER 
#
# Also include contig rmsd: 
#
#   ./calc_metrics.sh FOLDER TEMPLATE RECEPTOR 
#
# Also include contig and interface rmsd: 
#
#   ./calc_metrics.sh FOLDER TEMPLATE RECEPTOR INTERFACE_RES_FILE
#
# FOLDER should contain PDBs. INTERFACE_RES_FILE should contain space-delimited
# integers. This script will calculate metrics on every pdb and leave outputs
# in FOLDER.
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # abs path of this script

pre=`basename $1`
sbatch -J lddt.$pre $DIR/predict_lddt.sh $1
$DIR/calc_cce.sh $1 cce.$pre
sbatch -J frag.$pre $DIR/pick_frags_calc_qual.sh $1
if [ "$#" -eq 3 ]; then
    sbatch -J pymetric.$pre --mem=8g --wrap="$DIR/pymol_metrics.py --template $2 --receptor $3 $1"
elif [ "$#" -eq 4 ]; then
    sbatch -J pymetric.$pre --mem=8g --wrap="$DIR/pymol_metrics.py --template $2 --receptor $3 --interface_res $4 $1"
fi
