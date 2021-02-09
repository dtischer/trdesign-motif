#!/software/conda/envs/pyrosetta/bin/python3.7

'''
Simple utility script for removing N/C termini that don't
have secondary structure. Only meant to be called
on single chains
'''

from pyrosetta import *
from pyrosetta.rosetta import *
import argparse, sys, pickle, os
import numpy as np

sys.path.append('/home/dtischer/code/pyRosetta_utils/')
from pyR_utils import get_one_and_twobody_energies

init('-beta_nov16') 

###########################################
# Fucntions
###########################################
def get_few_contacts(pose, threshold=-0.4):
  '''
  Outputs
   - to_trim: boolean array. True if the residue makes few (non-local) contacts
  '''
  
  # get 2 bodies energies, excluding fa_rep
  sfxn_no_rep = core.scoring.ScoreFunctionFactory.create_score_function('beta_nov16')
  sfxn_no_rep.set_weight(core.scoring.fa_rep, 0.0)
  e1b, e2b = get_one_and_twobody_energies(pose, sfxn_no_rep)

  # mask to exclude the diagonal. we only want long range interactions
  u = np.triu(np.ones(e2b.shape), k=6)
  l = np.tril(np.ones(e2b.shape), k=-6)
  mask = (u+l) != 0
  e2b *= mask

  # rolling average (this is 0 indexed!!!)
  e2b_tot = e2b.sum(0)
  window_size = 5
  local_mean = np.convolve(e2b_tot, np.ones(window_size), mode='valid') / window_size
  local_mean_N2C = np.pad(local_mean, pad_width=(0,window_size-1), constant_values=(-2,-2))
  local_mean_C2N = np.pad(local_mean, pad_width=(window_size-1,0), constant_values=(-2,-2))
  local_mean = np.maximum(local_mean_N2C, local_mean_C2N)
  to_trim = local_mean > threshold
  
  return to_trim


def trim_tails(pose, trb_path=None):
  '''
  Remove sections fron N and C termini that are either loopy
  or do not make many contacts with the rest of the structure.
  --------------------------------------------------------
  Inputs
   - pose: single chain pose
  '''

  # calc DSSP

  
  # define the start/end of contigs.
  if trb_path is not None:
    with open(trb_path, 'rb') as infile:
      trb = pickle.load(infile)
    chs, idxs = zip(*trb['con_hal_pdb_idx'])
    con_idx_min = min(idxs)
    con_idx_max = max(idxs)
  else:
    con_idx_min = 10000
    con_idx_max = -10000

  # Initial calculation of dssp and to_trim
  DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()  # Thank you Chris Norn!
  DSSP.apply(pose)    # populates the pose's Pose.secstruct
  to_trim = get_few_contacts(pose)
  
  # Trim N-term
  while to_trim[0] or pose.secstruct()[pose.chain_begin(1) - 1 : pose.chain_begin(1) + 1] == 'LL':    
    # don't trim into the contigs
    pdb_idx, pdb_ch = pose.pdb_info().pose2pdb(pose.chain_begin(1)).split()
    pdb_idx = int(pdb_idx)
    if (pdb_idx >= con_idx_min) and (pdb_idx <= con_idx_max):
      break

    # trim if necesary
    pose.delete_residue_slow(pose.chain_begin(1))
    
    # update dssp and to_trim
    DSSP.apply(pose)
    to_trim = get_few_contacts(pose)
    
  # Trim C-term
  while to_trim[-1] or pose.secstruct()[pose.chain_end(1) - 3 : pose.chain_end(1) - 1] == 'LL':
    # don't trim into the contigs
    pdb_idx, pdb_ch = pose.pdb_info().pose2pdb(pose.chain_end(1)).split()
    pdb_idx = int(pdb_idx)
    if (pdb_idx >= con_idx_min) and (pdb_idx <= con_idx_max):
      break
    
    # trim if necesary
    pose.delete_residue_slow(pose.chain_end(1))
    
    # update dssp and to_trim
    DSSP.apply(pose)
    to_trim = get_few_contacts(pose)

  verbose = True
  if verbose:
    print('AFTER TRIMMING ENDS')
    print(pose.pdb_info())
    print(pose.secstruct())
    print(pose.sequence())
    
  return pose
  
  
#####################################################
# Main
#####################################################
if __name__ == "__main__":
  # Parse 'dem args
  parser = argparse.ArgumentParser()
  parser.add_argument('--pdb', nargs='+', required=True, help='pdb or space separated list of pdbs to trim')
  parser.add_argument('--trb_dir', help='dir to find the corresponding trb file if different than the pdb\'s dir')
  parser.add_argument('--suffix', help='suffix to add to dumped pdb', default='_trimmed')
  args = parser.parse_args() 
  
  # loop over pdbs
  for pdb_path in args.pdb:
    print('Trimming', pdb_path, '...')
    
    # Find the trb_dir
    if args.trb_dir is not None:
      trb_dir = args.trb_dir
    else:
      trb_dir = os.path.dirname(os.path.abspath(pdb_path))
    
    # Check if the trb file exists
    trb_putative = f'{trb_dir}/{os.path.basename(pdb_path)}'.replace('.pdb', '.trb')      
    trb_path = trb_putative if os.path.exists(trb_putative) else None
        
    # Trim the tails
    pose = pose_from_file(pdb_path)
    pose = trim_tails(pose, trb_path)
    pose.dump_pdb(pdb_path.replace('.pdb', f'{args.suffix}.pdb'))
