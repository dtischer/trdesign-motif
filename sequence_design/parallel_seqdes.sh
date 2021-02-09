#!/bin/bash
# Wrapper script for automatically submitting jobs for
# entire binder design pipeline of a directory of pdbs
# Does not do any heavy lifting itself. Okay to run 
# on the head node
# 210106 dt


##################
# Inputs
##################
# -t : Protein target you are trying to bind to. Typically a receptor.
# -b : Naturally occuring binding protein you are trying to make a mimetic of
# -r : Residues from the binder pdb whose identity you want to keep frozen during fast design.
#      Numbering is that in the pdb on chain A. Currently does not support chains other than chain A.
#      BE SURE TO SINGLE QUOTE FROZEN RESIDUES ex: -r '1 2 3 4 5'

# example usage (run from the directory with the .trb, .npz and .fas files):
# ~/projects/discon_motifs/scripts/parallel_fastdes.sh -t /home/dtischer/projects/discon_motifs/data/ref_struc/IL4_receptor_sc.pdb -b /home/dtischer/projects/discon_motifs/data/benchmarks/IL4/3BPL_xtl.pdb -r '5 6 8 9 11 12 13 15 78 81 82 85 87 88 89 91 92 114 115 118 121 122' 

##################
# Main
##################
GIT_ROOT=$(git rev-parse --show-toplevel)

# Parse inputs
while getopts l:o:t:b:r: flag
do
  case "${flag}" in
    t) target=${OPTARG};;
    b) binder=${OPTARG};;
    r) residues_2_keep=${OPTARG};;  # BE SURE TO SINGLE QUOTE FROZEN RESIDUES ex: -r '1 2 3 4 5'
  esac
done

# make the task file
for file in $PWD/*.trb; do
  bn=`echo $file | sed 's/.trb//'`
  cmd="$GIT_ROOT/sequence_design/mk_struc_pssm_design.sh "
  cmd+="-l $bn -t $target -b $binder -r '$residues_2_keep'"  # important that the residues remain in single quotes!
  echo $cmd
done >| tasks_fd.sh

# submit task array to slurm
TASK_FILE="tasks_fd.sh"
GROUP_SIZE=1
NUM_TASKS=$(cat $TASK_FILE|wc -l)
sbatch -a 1-$(($NUM_TASKS / $GROUP_SIZE + 1)) <<END
#!/bin/bash
#SBATCH -p short
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -o %A_%a.log

    #submit multiple lines of the task file in on slurm_array_task_id (takes on values of -a flag)
    for I in \$(seq 1 $GROUP_SIZE); do
        J=\$((\$SLURM_ARRAY_TASK_ID * $GROUP_SIZE + \$I - $GROUP_SIZE))
        CMD=\$(sed -n "\${J}p" $TASK_FILE )
        echo "\${CMD}" | bash
    done
END
