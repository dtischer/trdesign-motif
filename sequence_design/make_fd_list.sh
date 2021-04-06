#!/bin/bash
#
# Generates FastDesign commands for a folder of hallucinations and outputs to
# stdout. Run them using submit_fd.sh
#
# Usage (redirecting output to task list file): 
#
#   ./make_fd_list.sh FOLDER TEMPLATE_PDB RECEPTOR_PDB INTERFACE_RES_FILE > fd.list
#
# FOLDER should contain PDBs of structures you want to design, along with TRB
# files mapping their constrained regions to the template.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # folder of this script

mkdir -p out
mkdir -p pssm
mkdir -p $1/trimmed

for PDB in $1/*.pdb; do
    ID=`basename $PDB .pdb`
    TRB=`dirname $PDB`/$ID.trb
    cmd=""
    if [ ! -f $1/trimmed/$ID.pdb ]; then
        cmd="$DIR/trim_tails.py --pdb $PDB --suffix='' --out_dir $1/trimmed &>>$ID.fd.log; "
    fi
    TRIMMED=$1/trimmed/`basename $PDB`
    if [ ! -f pssm/${ID}_0001.profile.pssm ]; then
        cmd+="$DIR/pssm/mk_pssm.sh $TRIMMED &>>$ID.fd.log; "
    fi
    cmd+="$DIR/structure2design.py --pdb_in $TRIMMED --trb_file $TRB --out_dir $1 --native $2 --tar $3 --freeze_native_residues `cat $4` --pssm_file=pssm/${ID}_0001.profile.pssm --pssm_mode=norn1 --layer_design=True &>> $ID.fd.log"
    echo $cmd
done  

