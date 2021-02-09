#!/bin/bash -x
# use bash -x to run in tracing mode
source activate /software/conda/envs/tensorflow

# gen structure from npz, gen pssm, design with binder residues on and target in place

#E.G. bash gen_struc_pssm_design.sh -l /home/lisanza/ig_fold/discontinous_motifs/B7_2/ferredoxin_scaffold/ferredoxin_B7_2_11_16new_20_0_1_L8L5L13L5L3 -t '/home/lisanza/ig_fold/discontinous_motifs/input/CTLA4_relaxed.pdb' -b '/home/lisanza/ig_fold/discontinous_motifs/B7_2/B7_2_helix_overlay_renum.pdb' -r '79 77 74 65 63 61 24 22 20 35 33 31 72 67 68 69 70 28'

# BE SURE TO SINGLE QUOTE FROZEN RESIDUES ex: -r '1 2 3 4 5' AND that they appear this way in the tasks file!
# If writing the task file with a bash for loop, may need to do something like this:
# echo "-r '1 2 3 4 5'" > task

GIT_ROOT=$(git rev-parse --show-toplevel "$0")

# 0. Parse the args!
while getopts l:t:b:r: flag
do
	case "${flag}" in
        l) loc_npz_fasta_trb=${OPTARG};; #inlcude basename, but no file extension
        t) target=${OPTARG};;
        b) binder=${OPTARG};;
        r) residues_2_keep=${OPTARG};;
    esac
done

name=`echo ${loc_npz_fasta_trb} | awk -F '/' '{print $NF}'`
echo ${name}
x=$(readlink -f $loc_npz_fasta_trb)
BASE_DIR=$(dirname $x)  # how to make into one liner?

echo "The Git root is: ${GIT_ROOT} baaabb"

#1. generate structure:
mkdir -p ${BASE_DIR}/trR_models
#${GIT_ROOT}/sequence_design/trFolding/RosettaTR.py -m 0 -pd 0.15 ${loc_npz_fasta_trb}.npz ${loc_npz_fasta_trb}.fas ${BASE_DIR}/trR_models/${name}.pdb
${GIT_ROOT}/sequence_design/trim_tails.py --pdb ${BASE_DIR}/trR_models/${name}.pdb --suffix ''

exit

#2. generate PSSM
mkdir -p ${BASE_DIR}/out
mkdir -p ${BASE_DIR}/pssm
${GIT_ROOT}/sequence_design/pssm/mk_pssm.sh ${BASE_DIR}/trR_models/${name}.pdb

#3 design
${GIT_ROOT}/sequence_design/structure2design.py --pdb_in ${BASE_DIR}/trR_models/${name}.pdb --trb_file ${loc_npz_fasta_trb}.trb --pssm_file ${BASE_DIR}/pssm/${name}_0001.profile.pssm --freeze_native_residues ${residues_2_keep} --tar ${target}
