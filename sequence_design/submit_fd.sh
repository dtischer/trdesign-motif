#!/bin/bash
#
# Submits array job for Fastdesign. Generate the task list using
# make_fd_list.sh
#
# Usage:
#
#   ./submit_fd.sh fd.list
#

source activate pyrosetta

sbatch -a 1-$(cat $1 | wc -l) -J fastdes -c 1 --mem=10g \
       --wrap="eval \`sed -n \${SLURM_ARRAY_TASK_ID}p $1\`"

