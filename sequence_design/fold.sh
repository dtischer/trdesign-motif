#!/bin/bash
#
# Generates structural models of every .npz file in FOLDER.
#
# For trRosetta v1 outputs with 15-degree angle bins:
#
#   Usage: fold.sh FOLDER
#
# For trRosetta v2 outputs with 10-degree angle bins:
#
#   Usage: fold.sh -r FOLDER
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

OPTIND=1
while getopts ":r" opt; do
    case $opt in
        r) roll="--roll";;
    esac
done
shift $((OPTIND-1))

outdir=$1
task_file=`basename $1`.fold.list
for NPZ in $1/*.npz; do
  f=`basename $NPZ .npz`
  FASTA=`dirname $NPZ`/$f.fas
  if [ ! -f $outdir/$f.pdb ]; then
    echo "$DIR/trFolding2/RosettaTR.py $roll -m 0 -pd 0.15 $NPZ $FASTA $outdir/$f.pdb"
  fi
done > $task_file

count=$(cat $task_file | wc -l)
if [ "$count" -gt 0 ]; then
    echo "Folding $count designs..."
    sbatch -a 1-$(cat $task_file | wc -l) -J fold.`basename $1` -c 1 --mem=10g \
           -o /dev/null -e /dev/null \
           --wrap="eval \`sed -n \${SLURM_ARRAY_TASK_ID}p $task_file\`"
else
    echo "No designs need folding. To refold, delete or move existing .pdb files."
fi
