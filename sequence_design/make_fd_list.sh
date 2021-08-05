#!/bin/bash
#
# Generates FastDesign commands for a folder of hallucinations and outputs to
# stdout. Run them using submit_fd.sh. This does not generate commands for
# designs that already have outputs (in FOLDER/fast_designs/chA) so delete them
# if you're trying to repeat runs.
#
# Usage (redirecting output to task list file): 
#
#   ./make_fd_list.sh FOLDER TEMPLATE_PDB RECEPTOR_PDB INTERFACE_RES_FILE >
#   fd.list
#
# FOLDER should contain PDBs of structures you want to design, along with TRB
# files mapping their constrained regions to the template.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # folder of this script

mkdir -p $1/out
mkdir -p $1/pssm
mkdir -p $1/trimmed
mkdir -p $1/log

# absolute paths
ABS1=`readlink -f $1` 
ABS2=`readlink -f $2` 
ABS3=`readlink -f $3` 
ABS4=`readlink -f $4` 

for PDB in $1/*.pdb; do
    ID=`basename $PDB .pdb`
    TRB=$ID.trb
    logfile=log/$ID.fd.log

    cmd="cd $ABS1; "
    if [ ! -f $1/trimmed/$ID.pdb ]; then
        cmd+="$DIR/trim_tails.py --pdb `basename $PDB` --suffix='' --out_dir trimmed/ &>> $logfile; "
    fi
    TRIMMED=trimmed/`basename $PDB`
    if [ ! -f $1/pssm/${ID}_0001.profile.pssm ]; then
        cmd+="$DIR/pssm/mk_pssm.sh $TRIMMED &>> $logfile; "
    fi
    if [ ! -f $1/fast_designs/chA/$ID.pdb ]; then
        if [[ "$ABS3" == *"None" ]]; then
            cmd+="$DIR/structure2design.py --pdb_in $TRIMMED --trb_file $TRB --out_dir ./ --native $ABS2 --freeze_native_residues `cat $ABS4` --pssm_file=pssm/${ID}_0001.profile.pssm --pssm_mode=norn1 --layer_design=True &>> $logfile"
        else
            cmd+="$DIR/structure2design.py --pdb_in $TRIMMED --trb_file $TRB --out_dir ./ --native $ABS2 --tar $ABS3 --freeze_native_residues `cat $ABS4` --pssm_file=pssm/${ID}_0001.profile.pssm --pssm_mode=norn1 --layer_design=True &>> $logfile"
        fi
        echo $cmd
    fi
done  

