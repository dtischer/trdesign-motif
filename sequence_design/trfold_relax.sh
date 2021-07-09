#!/bin/bash
#
# Performs relax on TRFold structures.
#
# Usage:
#
#   ./trfold_relax.sh FOLDER
#
# will relax all pdb files in folders, using restraints derived from matching
# npz files, and output relaxed structures into a subfolder called trf_relax/.
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

OPTIND=1
roll="--roll"
while getopts ":n" opt; do
    case $opt in
        n) roll="";;
    esac
done
shift $((OPTIND-1))

outdir=$1/trf_relax
mkdir -p $outdir
task_file=`basename $1`.fold.list
for NPZ in $1/*.npz; do
  f=`basename $NPZ .npz`
  PDB=`dirname $NPZ`/$f.pdb
  if [ ! -f $outdir/$f.pdb ]; then
    echo "$DIR/trfold_relax.py $roll -sg 7,3 $NPZ $PDB $outdir/$f.pdb"
  fi
done > $task_file

count=$(cat $task_file | wc -l)
if [ "$count" -gt 0 ]; then
    echo "Relaxing $count designs..."
    sbatch -a 1-$(cat $task_file | wc -l) -J fold.`basename $1` -c 1 -p short --mem=4g \
           -o /dev/null -e /dev/null \
           --wrap="eval \`sed -n \${SLURM_ARRAY_TASK_ID}p $task_file\`"
else
    echo "No designs need relaxing. To refold, delete or move existing .pdb files."
fi
