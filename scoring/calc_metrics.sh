#!/bin/bash
#
# Calculate lddt, rmsd, ss frac, and parses losses:
#
#   ./calc_metrics.sh FOLDER 
#
# "Full" mode (also does cce10, avg_all_frags):
#
#   ./calc_metrics.sh -f FOLDER
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

OPTIND=1
simple=true
while getopts ":sf" opt; do
    case $opt in
        s) simple=true;;
        f) simple=false;;
    esac
done
shift $((OPTIND-1))

pre=`basename $1`
sbatch -J lddt.$pre $DIR/predict_lddt.sh $1
sbatch --mem 1g -J pyros.$pre --wrap="$DIR/pyrosetta_metrics.py $1"

if [ $simple = "false" ]; then
    $DIR/get_cce.sh $1 cce.$pre
    sbatch -J frag.$pre $DIR/pick_frags_calc_qual.sh $1
    $DIR/get_trr_rmsd.sh $1
    $DIR/get_trunk_rmsd.sh $1
    sbatch -J bcov_metrics.$pre -c 8 --mem 8g --wrap="$DIR/get_bcov_metrics.py $1/../complex/"
fi

if [ "$#" -eq 2 ]; then
    sbatch -J tmscore.$pre -c 1 --mem=2g --wrap="$DIR/get_tmscores.py --template $2 $1"
elif [ "$#" -eq 4 ]; then
    sbatch -J tmscore.$pre -c 1 --mem=2g --wrap="$DIR/get_tmscores.py --template $2 $1"
    sbatch -J pymetric.$pre --mem=8g --wrap="$DIR/pymol_metrics.py --template $2 --receptor $3 --interface_res $4 $1"
fi
