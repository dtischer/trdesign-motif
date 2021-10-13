#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # folder of this script

#get pssm
#run in directory with pssm and out directories present
# arg 1 pdb loc
# arg 2 
name=`echo ${1} | awk -F '/' '{print $NF}' | awk -F '.pdb' '{print $1}'`
echo ${name}

#get counts
/home/norn/Rosetta_PSSM/main/source/bin/rosetta_scripts.hdf5.linuxgccrelease -s ${1} @${DIR}/flags_make_pssm -parser:protocol ${DIR}/pssm_from_frags.xml

#make pssm from count file
${DIR}/mk_pssm_from_counts_file.py --pseudoCountWeight 50 --eff_count_method 2 --countAA out/${name}_0001.profile --pssmOut pssm/${name}_0001.profile.pssm
