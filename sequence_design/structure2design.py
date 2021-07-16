#!/software/conda/envs/pyrosetta/bin/python3.7

import sys,os,json,pickle,argparse
import tempfile
import numpy as np
import pandas as pd
import pandas
from optparse import OptionParser
import time
import glob
from scipy.interpolate import CubicSpline
from Bio import SeqIO

from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

import pyrosetta
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score
import pickle

sys.path.append('/home/dtischer/code/pyRosetta_utils/')
import hbond_utils as hbu  # for some reason I get an "UnboundLocalError" if I don't alias. But it's in globals() !
import design_utils 

os.environ["OPENBLAS_NUM_THREADS"] = "1"
script_dir = os.path.dirname(os.path.realpath(__file__))

def get_xyz(p, atom_id='CA'):
  '''
  Get the cartesian coordinates of an atom type for every residue
  in the pose. Atom ids only make sense if they are in the backbone.
  '''
  xyz = [p.residue(i).atom(atom_id).xyz() for i in range(1, p.size()+1)]
  xyz = np.stack(xyz, axis=0)
  
  return xyz


def main():
    t0 = time.time()

    ###############################################################################
    # Option parser
    ###############################################################################
    parser = argparse.ArgumentParser()
    
    # Pose
    parser.add_argument('--pdb_in', type=str, help='Path to input pdb file')
    parser.add_argument('--out_dir', type=str, help='Path to output directory (if different that the pdb dir)')
    parser.add_argument('--trb_file', type=str, help='tracker file for min loss step of hallucination')
    parser.add_argument('--pssm_file', type=str, help='path to pssm file')
    
    # Force aa from contigs
    parser.add_argument('--freeze_native_residues', type=int, nargs='+', help='force these aa from the native into the hallucinated protein. PDB residue index.')
    parser.add_argument('--frozen_chain', type=str, default='A', help='native chain the frozen residues are from')
    parser.add_argument('--native', type=str, help='Path to the native PDB (where the contig geometry was taken from).'
                       'This overrides any option in trb["settings"]["pdb"]')
    
    # Target of binding (usually a receptor)
    parser.add_argument('--tar', type=str, help='Path to target pdb file')
    
    # Sequence design options
    parser.add_argument('--sfxn', type=str, default='beta_nov16', help='What score function to use <beta_nov16, HH_19>')
    parser.add_argument('--pssm_mode', type=str, default='norn1', help='Method to generate and weight pssm')
    parser.add_argument('--layer_design', type=bool, default=False, help='Use layer design?')
    
    args = parser.parse_args()
    
    ###############################################################################
    # Init pyrosetta
    ###############################################################################
    # sequence design options
    if args.pssm_mode == 'norn1':
      min_aa_probability, sequnce_profile_weight = -0.125, 1.75   # cutoff for packer, change bias in score function
    elif args.pssm_mode == 'norn2':
      min_aa_probability, sequnce_profile_weight = -0.5, 1
    elif args.pssm_mode == 'trR':
      min_aa_probability, sequnce_profile_weight = -0.5, 0.3
    
    # load hal meta data
    with open(args.trb_file, 'rb') as infile:
        trb = pickle.load(infile)
        
    # Resolve the output directory
    if args.out_dir is None:
        out_dir = os.path.dirname(args.pdb_in)
    else:
        out_dir = args.out_dir
        
    # Make needed dirs
    os.makedirs(out_dir+'/fast_designs', exist_ok=True) 
    os.makedirs(out_dir+'/fast_designs/complex', exist_ok=True)
    os.makedirs(out_dir+'/fast_designs/chA', exist_ok=True)
    os.makedirs(out_dir+'/fast_designs/hb_stats', exist_ok=True)
        
    # Resolve path to the native pdb
    if args.native is not None:
        nat_f = args.native
    elif ('settings' in trb) and ('pdb' in trb['settings']):
        nat_f = trb['settings']['pdb']
    else:
        err = 'Error: A path to the native pdb must be provided either as "--native" or as trb["settings"]["pdb"]'
        sys.exit(err)
        
    # Resolve path to score function weights    
    if args.sfxn == 'beta_nov16':
      weight_file = f'{script_dir}/protocols/beta16_nostab.wts'
      flag_file = f'{script_dir}/protocols/beta16_nostab.flg'
    elif args.sfxn == 'HH_19':
      weight_file = f'{script_dir}/protocols/HH_run19A_weights_266.wts'
      flag_file = f'{script_dir}/protocols/HH_run19A_weights_266.flg'
    else:
      err = 'Please provide a valid option for the score function to use'
      sys.exit(err)
        
    # init pyR
    init(f'@{flag_file} -holes:dalphaball /home/norn/software/DAlpahBall/DAlphaBall.gcc')
    
    # Design basename
    bn_des = args.pdb_in.split('/')[-1].replace('.pdb', '')
    
    # Load pdbs
    p_hal = pyrosetta.pose_from_file(args.pdb_in)
    p_nat = pose_from_file(nat_f)
    p_tar = pose_from_file(args.tar)
    info = p_hal.pdb_info()
    
    ###############################################################################
    # Doug's attempt to pare down extracting hbnet constraints to a minimal form
    ###############################################################################    
    # get info on hb in native ptn
    df_nat_hb = hbu.compute_hbond_energies(nat_f)  # all indexing is pose indexing, not pdb indexing
    
    # grab pose index of contigs (pdb index may not start at 1 because of trimming!)
    con_nat_pose_idx = [p_nat.pdb_info().pdb2pose(*i) for i in trb['con_ref_pdb_idx']]
    con_hal_pose_idx = [p_hal.pdb_info().pdb2pose(*i) for i in trb['con_hal_pdb_idx']]
    
    for i,j in zip(con_nat_pose_idx, con_hal_pose_idx):
        print(i,j)
    
    # dicts for easy interconversion
    nat2hal_pose_idx = dict(zip(con_nat_pose_idx, con_hal_pose_idx))
    hal2nat_pose_idx = dict(zip(con_hal_pose_idx, con_nat_pose_idx))
    
    # narrow down to hb that have acc/don in contig of nat
    m = df_nat_hb['acc_res_n'].isin(con_nat_pose_idx) & df_nat_hb['don_res_n'].isin(con_nat_pose_idx)
    df_nat_hb = df_nat_hb[m]
    
    # grab the aa of nat hb res (or bb if hb does not involve sc)
    # 1. df of all res and atom types involved in hb in nat contig
    df_acc = df_nat_hb[m][['acc_res_n', 'acc_res_name', 'acc_atom_type_name']]
    df_acc.rename(lambda x: x[4:], axis='columns', inplace=True)
    df_don = df_nat_hb[m][['don_res_n', 'don_res_name', 'don_atom_type_name']]
    df_don.rename(lambda x: x[4:], axis='columns', inplace=True)
    df_con_hb = pd.concat([df_acc, df_don], axis=0)
    
    # 2. For each res, are all hb atoms in the backbone?
    df_con_hb['bb'] = df_con_hb['atom_type_name'].str.contains('bb')
    bb = df_con_hb[['res_n', 'bb']].groupby(by='res_n').all()    # are all hb to the res bb contacts?
    aa3 = df_con_hb[['res_n', 'res_name']].groupby(by='res_n').min()  # just get the aa3 of each res
    hb_by_res = pd.concat([aa3, bb], axis=1)
    
    # 3. Force the native aa only if it's sc is involved in an hb
    hb_by_res['aa_2_force'] = hb_by_res.apply(lambda x: 'bb' if x['bb'] else x['res_name'], axis=1)
    
    # pack into one df (each row is an hbond in the nat that is fully contained in the contigs)
    df_selected_hbs = df_nat_hb[m]
    df_selected_hbs['acc_res_hal'] = df_selected_hbs.apply(lambda x: nat2hal_pose_idx[x['acc_res_n']], axis=1)  # record equivalent hal res
    df_selected_hbs['acc_2_force'] = hb_by_res.loc[df_selected_hbs['acc_res_n']]['aa_2_force'].values  # pack the acc as nat aa or bb?
    df_selected_hbs['don_res_hal'] = df_selected_hbs.apply(lambda x: nat2hal_pose_idx[x['don_res_n']], axis=1)  # record equivalent hal res
    df_selected_hbs['don_2_force'] = hb_by_res.loc[df_selected_hbs['don_res_n']]['aa_2_force'].values  # pack the don as nat aa or bb?
    
    # make chris-like df where each row is an entire hbnetwork
    df_selected_csts = pd.DataFrame()
    df_selected_csts['cst_idxes'] = [np.array([nat2hal_pose_idx[i] for i in hb_by_res.index])]  # wrap in list so that array isn't "unpacked"
    df_selected_csts['csts_type'] = 'hbnet'
    df_selected_csts['hal_idxes'] = [np.array([nat2hal_pose_idx[i] for i in hb_by_res.index])]
    df_selected_csts['member_size'] = len(hb_by_res)
    df_selected_csts['native_id'] = nat_f
    df_selected_csts['native_idxes'] = [np.array((hb_by_res.index))]
    df_selected_csts['setid'] = 'setid'
    df_selected_csts['set_idx'] = 0
    df_selected_csts['seq'] = [np.array([design_utils.aa321.get(aa3, 'bb') for aa3 in hb_by_res['aa_2_force']])]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 100000):
        print(df_selected_csts)

    ###############################################################################
    # Protocol
    ###############################################################################
    def mk_xml(weight_file, min_aa_probability, sequnce_profile_weight, layer_design, pssm_f_updated, hbnet_resnums_str, cart=False):
      pack_protocol = f"""
      <ROSETTASCRIPTS>
        <SCOREFXNS>
              <ScoreFunction name="sfxn_pure" weights="{weight_file}">
                  {'<Reweight scoretype="cart_bonded" weight="0.5"/>' if cart else ''}
                  {'<Reweight scoretype="pro_close" weight="0.0"/>' if cart else ''}
              </ScoreFunction>

              <ScoreFunction name="SFXN3" weights="{weight_file}">  
                  <Reweight scoretype="res_type_constraint" weight="{sequnce_profile_weight}"/>  # experiment with this value original designs from 3_25_20 were 0.3
                  <Reweight scoretype="hbond_lr_bb" weight="2"/>            #primarily affects beta sheets
                  <Reweight scoretype="hbond_sr_bb" weight="1.5"/>      # primarily affect alpha helix
                  <Reweight scoretype="aa_composition" weight="1.0" />                                                       # composition constraint mover
                  Reweight scoretype="omega" weight="5"/>                           # penalty on omega bb angle. check if omega distribution is near wt or not. if not, add in
                  <Reweight scoretype="atom_pair_constraint" weight="1.0" />
                  <Reweight scoretype="coordinate_constraint" weight="1.0"/>
                  
                  <Reweight scoretype="approximate_buried_unsat_penalty" weight="5.0" />  # ask bcov what is current state of the art
                  <Set approximate_buried_unsat_penalty_assume_const_backbone="true" />
                  <Set approximate_buried_unsat_penalty_natural_corrections1="true" />
                  <Set approximate_buried_unsat_penalty_hbond_energy_threshold="-1.0" />
                  <Set approximate_buried_unsat_penalty_hbond_bonus_cross_chain="-2.5" />
                  <Set approximate_buried_unsat_penalty_hbond_bonus_ser_to_helix_bb="0.0" />
                  
                  {'<Reweight scoretype="cart_bonded" weight="0.5"/>' if cart else ''}
                  {'<Reweight scoretype="pro_close" weight="0.0"/>' if cart else ''}                  
                  
              </ScoreFunction>
        </SCOREFXNS>

        <RESIDUE_SELECTORS>
            <Layer name="init_core_SCN" select_core="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Layer name="init_boundary_SCN" select_boundary="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Layer name="surface_SCN" select_surface="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Index name="hbnet_res" resnums="{hbnet_resnums_str}"/>
            <Not name="not_hbnet_res" selector="hbnet_res" />
            <And name="surface_SCN_and_not_hbnet_res" selectors="surface_SCN,not_hbnet_res"/>

            <Chain name="chainA" chains="1" />
            <Chain name="chainB" chains="2" />
            <InterfaceByVector name="interface" grp1_selector="chainA" grp2_selector="chainB" />
            <And name="interface_chainB" selectors="interface,chainB" />
            <Not name="not_interface" selector="interface" />
            <And name="not_interface_chainB" selectors="not_interface,chainB" />

            <ResiduePDBInfoHasLabel name="HOTSPOT_res" property="HOTSPOT" />

            # Layer design selectors
            <SecondaryStructure name="strands_ini" ss="E" include_terminal_loops="0"  use_dssp="1" />
            <SecondaryStructure name="loops_ini" ss="L" include_terminal_loops="0"  use_dssp="1" />
            <PrimarySequenceNeighborhood name="loops_edges_ini" lower="1" upper="1" selector="loops_ini" />
            <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
            <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
            <And name="strands" selectors="strands_ini" />
            <And name="loops" selectors="loops_ini" />
            <And name="loops_and_edges" selectors="loops_edges_ini" />
            <And name="edges" selectors="strands,loops_and_edges"/>
            <Not name="strand_secondary_structure_ini" selector="loops_and_edges" />
            <And name="strand_secondary_structure" selectors="strand_secondary_structure_ini" />
            
            <Layer name="surface" select_surface="1" use_sidechain_neighbors="1"
                           core_cutoff="3.0" surface_cutoff="1.8" />
            <Layer name="boundary" select_boundary="1" use_sidechain_neighbors="1"
                           core_cutoff="3.0" surface_cutoff="1.8" />
            <Layer name="core" select_core="1" use_sidechain_neighbors="1"
                           core_cutoff="3.0" surface_cutoff="1.8" />
            <Layer name="all_layers" select_core="1" select_boundary="1" select_surface="1" use_sidechain_neighbors="1" core_cutoff="3.0" surface_cutoff="1.8" />

            # strand and loop
            <And name="strand_layers" selectors="strand_secondary_structure,all_layers" />
            <And name="strand_surface" selectors="strand_secondary_structure,surface" />
            <And name="strand_boundary" selectors="strand_secondary_structure,boundary" />
            <And name="strand_core" selectors="strand_secondary_structure,core" />
            <And name="loops_and_edges_surface" selectors="loops_and_edges,surface" />
            <And name="loops_and_edges_boundary" selectors="loops_and_edges,boundary" />
            <And name="loops_and_edges_core" selectors="loops_and_edges,core" />
            <And name="edges_core" selectors="edges,core" />
            <And name="loops_core" selectors="loops,core" />
            <And name="loops_not_resfile" selectors="loops_and_edges" />
            <Not name="not_loops" selector="loops_and_edges" />

            # helix
            <And name="helix_cap" selectors="entire_loop">
                <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
            </And>
            <And name="helix_start" selectors="entire_helix">
                <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
            </And>
                <And name="helix" selectors="entire_helix">
                <Not selector="helix_start"/>
            </And>
            <And name="helix_tail" selectors="entire_loop">
                <PrimarySequenceNeighborhood lower="0" upper="1" selector="entire_helix"/>
            </And>
            <And name="helix_end" selectors="entire_helix">
                <PrimarySequenceNeighborhood lower="1" upper="0" selector="helix_tail"/>
            </And>
            <And name="surface_helix_start" selectors="surface,helix_start" />
            <And name="surface_helix" selectors="surface,helix" />

            <And name="boundary_helix_start" selectors="boundary,helix_start" />
            <And name="boundary_helix" selectors="boundary,helix" />

            <And name="core_helix_start" selectors="core,helix_start" />
            <And name="core_helix" selectors="core,helix" />
        </RESIDUE_SELECTORS>

        <TASKOPERATIONS>
            <InitializeFromCommandline name="init" />
            <SeqprofConsensus name="pssm_cutoff" filename="{pssm_f_updated}" min_aa_probability="{min_aa_probability}" convert_scores_to_probabilities="0" probability_larger_than_current="0" debug="1" ignore_pose_profile_length_mismatch="1"/>  # restricts aa the packer can use based on a pssm.
            <RestrictAbsentCanonicalAAS name="noCys" keep_aas="ADEFGHIKLMNPQRSTVWY"/>
            <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
            <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True"/>
            <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1"/>
            <IncludeCurrent name="ic"/>
            <RestrictResiduesToRepacking name="fix_hbnet_residues" residues="{hbnet_resnums_str}"/>

            <OperateOnResidueSubset name="ld_surface_not_hbnets" selector="surface_SCN_and_not_hbnet_res">
                <RestrictAbsentCanonicalAASRLT aas="EDHKRQNSTPG"/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="repack_hotspots" selector="HOTSPOT_res">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="repack_interface_chainB" selector="interface_chainB">
                <RestrictToRepackingRLT/>
            </OperateOnResidueSubset>

            <OperateOnResidueSubset name="freeze_not_interface_chainB" selector="not_interface_chainB">
                <PreventRepackingRLT/>
            </OperateOnResidueSubset>
            
            # layer design task ops
            # strand and loops
            <OperateOnResidueSubset name="design_loops_surface" selector="loops_and_edges_surface">
                    <RestrictAbsentCanonicalAASRLT aas="DEGKNPQRST"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="design_loops_boundary" selector="loops_and_edges_boundary">
                    <RestrictAbsentCanonicalAASRLT aas="ADEGKNPQRSTV"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="design_loops_core" selector="loops_core">
                    <RestrictAbsentCanonicalAASRLT aas="AGILPV"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="design_edges_core" selector="edges_core">
                    <RestrictAbsentCanonicalAASRLT aas="AGPVILMWFY"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="strand_surface_aa" selector="strand_surface">
                    <RestrictAbsentCanonicalAASRLT aas="EHKRQST"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="strand_boundary_aa" selector="strand_boundary">
                    <RestrictAbsentCanonicalAASRLT aas="EFIKLQRSTVWY"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="strand_core_aa" selector="strand_core">
                    <RestrictAbsentCanonicalAASRLT aas="VILMWFY"/>
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="design_core" selector="core">
                    <RestrictAbsentCanonicalAASRLT aas="VILMWFYAGP"/>
            </OperateOnResidueSubset>

            # helix:
            <OperateOnResidueSubset name="ld1" selector="surface_helix_start" >
                <RestrictAbsentCanonicalAASRLT aas="DEHKPQR" />
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="ld2" selector="surface_helix" >
                <RestrictAbsentCanonicalAASRLT aas="EHKQR" />
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="ld5" selector="boundary_helix_start" >
                <RestrictAbsentCanonicalAASRLT aas="ADEHIKLMNPQRSTVWY" />
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="ld6" selector="boundary_helix" >
                <RestrictAbsentCanonicalAASRLT aas="ADEFHIKLMNQRSTVWY" />
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="ld9" selector="core_helix_start" >
                <RestrictAbsentCanonicalAASRLT aas="AFILMPVWY" />
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="ld10" selector="core_helix" >
                <RestrictAbsentCanonicalAASRLT aas="AFILMVWY" />
            </OperateOnResidueSubset>
            <OperateOnResidueSubset name="ld13" selector="helix_cap" >
                <RestrictAbsentCanonicalAASRLT aas="DNST" />
            </OperateOnResidueSubset>
        </TASKOPERATIONS>

        <MOVERS>
          <FavorSequenceProfile name="FSP" scaling="none" weight="1.0" pssm="{pssm_f_updated}" scorefxns="SFXN3" chain="1"/>  # interacts with res type constraint

          <SwitchResidueTypeSetMover name="to_fullatom" set="fa_standard"/>

          <FastRelax name="fastRelax" scorefxn="sfxn_pure" task_operations="init,ex1_ex2aro,ic" cartesian="{str(cart).lower()}">
              <MoveMap name="MM">                
                  <ResidueSelector selector="chainA" chi="true" bb="true" bondangle="false" bondlength="false" />
                  <ResidueSelector selector="interface_chainB" chi="true" bb="false" bondangle="false" bondlength="false" />
                  <ResidueSelector selector="not_interface_chainB" chi="false" bb="false" bondangle="false" bondlength="false" />
                  <Jump number="1" setting="true" />
              </MoveMap>
          </FastRelax>

          <ClearConstraintsMover name="rm_csts" />

          <FastDesign name="fastDesign" scorefxn="SFXN3" repeats="2"  task_operations="init,ex1_ex2aro,ld_surface_not_hbnets,fix_hbnet_residues,ic,limitchi2,pssm_cutoff,noCys,repack_hotspots,repack_interface_chainB,freeze_not_interface_chainB{",strand_surface_aa,strand_boundary_aa,strand_core_aa,design_loops_surface,design_loops_boundary,design_loops_core,design_edges_core,ld1,ld2,ld5,ld6,ld9,ld10,ld13" if layer_design else ""}" batch="false" ramp_down_constraints="false" cartesian="{str(cart).lower()}" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"> 
              <MoveMap name="MM">                
                  <ResidueSelector selector="chainA" chi="true" bb="true" bondangle="false" bondlength="false" />
                  <ResidueSelector selector="interface_chainB" chi="true" bb="false" bondangle="false" bondlength="false" />
                  <ResidueSelector selector="not_interface_chainB" chi="false" bb="false" bondangle="false" bondlength="false" />
                  <Jump number="1" setting="true" />
              </MoveMap>
          </FastDesign>

          <SwitchChainOrder name="chain1onlypre" chain_order="1" />
          <ScoreMover name="scorepose" scorefxn="sfxn_pure" verbose="false" />
          <ParsedProtocol name="chain1only">  # deletes everything BUT chain 1 (A) and then scores it. (Note: It does not alter the pose after it exits)
              <Add mover="chain1onlypre" />
              <Add mover="scorepose" />
          </ParsedProtocol>

        </MOVERS>

        <FILTERS>
          <BuriedUnsatHbonds name="vbuns_all_heavy" report_all_heavy_atom_unsats="true" scorefxn="sfxn_pure" ignore_surface_res="false" print_out_info_to_pdb="true" atomic_depth_selection="5.5" burial_cutoff="1000" confidence="0" />
          <BuriedUnsatHbonds name="sbuns_all_heavy" report_all_heavy_atom_unsats="true" scorefxn="sfxn_pure" cutoff="4" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" atomic_depth_selection="5.5" atomic_depth_deeper_than="false" confidence="0" />

          # dt added filters
          <ScoreType name="score" scorefxn="sfxn_pure" score_type="total_score" threshold="0.0" confidence="0" />
          <MoveBeforeFilter name="score_monomer" mover="chain1only" filter="score" confidence="0" />

          <ResidueCount name="nres" confidence="0" />
          <MoveBeforeFilter name="nres_monomer" mover="chain1only" filter="nres" confidence="0" />

          <CalculatorFilter name="score_res_monomer" confidence="0" equation="SCORE/NRES" threshold="-2.1">
            <VAR name="SCORE" filter_name="score_monomer" />
            <VAR name="NRES" filter_name="nres_monomer" />
          </CalculatorFilter>

          <Geometry name="geom" count_bad_residues="true" confidence="0"/>
          <MoveBeforeFilter name="geom_monomer" mover="chain1only" filter="geom" confidence="0" />

          <Ddg name="ddg"  threshold="-10" jump="1" repeats="5" repack="1" confidence="0" scorefxn="sfxn_pure" extreme_value_removal="true"/>
        </FILTERS>

        <PROTOCOLS>
             <Add mover="FSP"/>
             add one round of fastDesign with npz constraints here
             <Add mover="fastDesign"/>
             <Add mover="rm_csts"/>
             <Add mover="fastRelax"/>

             <Add filter="vbuns_all_heavy"/>   #Very buns
             <Add filter="sbuns_all_heavy"/>    #Surface buns. "Almost doesn't matter" -Chirs Norn

             # dt added filters
             <Add filter_name="score_res_monomer" />
             <Add filter_name="geom_monomer"/>
             <Add filter_name="ddg" />

        </PROTOCOLS>

      </ROSETTASCRIPTS>
      """
      return pack_protocol
    
    ############################################################################################
    # Setup atom pair constraints for the hbonds
    ###########################################################################################    
    # Setting residue type constrained residues to their target identities
    info = p_hal.pdb_info()
    for i_row,row in df_selected_csts.iterrows():
        native_seq = row['seq']
        cst_idxes1 = row['hal_idxes']
        for resi1, aa in zip(cst_idxes1, native_seq):
            if (aa == 'bb') and (p_hal.residue(resi1).name() == 'PRO'):  # PRO can't accept bb hbonds
                aa = 'A'
            elif (aa == 'bb') and (p_hal.residue(resi1).name() != 'PRO'):
                continue
            resn = design_utils.aa123[aa]
            mutator = rosetta.protocols.simple_moves.MutateResidue(resi1, resn)
            mutator.apply(p_hal)
            info.add_reslabel(resi1,'csts_type_'+row['csts_type'])
            print(f'mutating {resi1} to {resn}')
    
    # Add pairwise distance csts for donor/acceptor groups
    hbondable_res_types = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'TYR', 'TRP']
    bb_hbondable_atom_types = ['H', 'O'] # don't constrain aa types, if it is the bb that donates/accepts
    hbond_csts_standard_dev = 0.25
    hbnet_resnums = []
    d_source_networks = {'don_res_n_design': [], 'acc_res_n_design':[], 'acc_res_n_native':[], 'don_res_n_native':[], 'energy_hbond_plus_elec':[], 'native_id':[], 'category':[]}
    for i_row,row in df_selected_csts.iterrows():
        native_id = row['native_id']
        native_idxes = row['native_idxes']
        hal_idxes1 = row['hal_idxes']
        hbnet_type = row['csts_type']
        
        #df_nat_hbonds = pd.read_csv(f'{opts.folder_of_hbond_annotations}/{native_id}.hbonds.txt') # 
        df_nat_hbonds = df_selected_hbs
        
        for resi_hal, resi_nat in zip(hal_idxes1, native_idxes):
            for resj_hal, resj_nat in zip(hal_idxes1, native_idxes):
                df_nat_hbond = df_nat_hbonds[(df_nat_hbonds['acc_res_n']==resi_nat) & (df_nat_hbonds['don_res_n']==resj_nat)]
                if len(df_nat_hbond)==0:
                    continue
                for i_bond, bond in df_nat_hbond.iterrows():
                    HAdist = bond['HAdist']
                    # print('native hbond')
                    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 10000):
                    #     print(df_nat_hbond)
                    
                    # Avoid hbnet csts from native hbonds that involve terminal don/acc types
                    don_atom_name = bond['don_atom_name']
                    acc_atom_name = bond['acc_atom_name']
                    don_res_name = bond['don_res_name']
                    acc_res_name = bond['acc_res_name']
                    terminal_atom_types = ['1H','2H','3H','OXT']
                    if (don_atom_name.strip() in terminal_atom_types) or (acc_atom_name.strip() in terminal_atom_types):
                        continue
                    
                    # By default Rosetta will mutate to the histidine tautomer with HE2. 
                    don_name = bond['don_res_name']
                    if don_name=='HIS':
                         don_atom_name = 'HE2'
                    
                    acc_res_no = bond['acc_res_n']
                    don_res_no = bond['don_res_n']
                    print(f"Setting up distance csts for hbond between ACCEPTOR: {acc_res_name} {resi_hal} atom {acc_atom_name}, and DONOR: {don_res_name} {resj_hal} atom {don_atom_name}")

                    id_acc = rosetta.core.id.AtomID(p_hal.residue(resi_hal).atom_index(acc_atom_name), resi_hal)
                    id_don = rosetta.core.id.AtomID(p_hal.residue(resj_hal).atom_index(don_atom_name), resj_hal)
                    func = rosetta.core.scoring.func.HarmonicFunc(HAdist, hbond_csts_standard_dev)
                    cstAB = rosetta.core.scoring.constraints.AtomPairConstraint(id_don, id_acc, func)
                    label = p_hal.add_constraint(cstAB)

                    # Only constrain identities for the residue types that can hbond
                    if acc_res_name in hbondable_res_types and acc_atom_name not in bb_hbondable_atom_types:
                        hbnet_resnums.append(int(resi_hal))
                    if don_res_name in hbondable_res_types and don_atom_name not in bb_hbondable_atom_types:
                        hbnet_resnums.append(int(resj_hal))

                    d_source_networks['acc_res_n_design'].append(resi_hal)
                    d_source_networks['don_res_n_design'].append(resj_hal)

                    d_source_networks['acc_res_n_native'].append(resi_nat+1)
                    d_source_networks['don_res_n_native'].append(resj_nat+1)

                    d_source_networks['energy_hbond_plus_elec'].append(bond['energy_hbond_plus_elec'])
                    d_source_networks['native_id'].append(native_id)
                    d_source_networks['category'].append(hbnet_type)
    
    source_networks = pd.DataFrame.from_dict(d_source_networks)
    source_networks.to_csv(f'{out_dir}/fast_designs/hb_stats/{bn_des}.beforeDesign.csv')
    
    print("Just after adding constraints")
    print(p_hal.constraint_set())
    
    ############################################################################################
    # Reformat the PSSM file. We need to do this as will otherwise restrict
    # hbnet amino acid rotamers at hbnet positions
    ###########################################################################################
    hbnet_resnums = np.sort(np.array(list(set(hbnet_resnums))))
    print('Forcing identities for residues', hbnet_resnums)
    pssm_f = args.pssm_file
    os.makedirs(f'{out_dir}/fast_designs/pssm', exist_ok=True)
    pssm_f_updated = f'{out_dir}/fast_designs/pssm/{bn_des}_updated.pssm' 
    pssm = design_utils.read_pssm(pssm_f)
    pssm[hbnet_resnums-1,:] = 0 # ResNums are 1 numbered
    
    ###########################################################################################
    # Design in the context of the binding target (ie - receptor)
    ###########################################################################################
    # dicts for easy interconversion of pdb_idx (ie - ('A', 7))
    nat2hal_pdb_idx = dict(zip(trb['con_ref_pdb_idx'], trb['con_hal_pdb_idx']))
    hal2nat_pdb_idx = dict(zip(trb['con_hal_pdb_idx'], trb['con_ref_pdb_idx']))

    # 1. mutate all CYS to ALA in p_hal
    for i,a in enumerate(p_hal.sequence()):
        if a == 'C':
            mutator = rosetta.protocols.simple_moves.MutateResidue(i+1,'ALA')
            mutator.apply(p_hal)
            print(f'mutating C{i+1}A')

    # 2. Force aa at specified positions along the contig
    for pdb_idx_nat in args.freeze_native_residues:
        # Check frozen residue is in the contig. Sometimes we accidently overspecify the frozen residues
        if (args.frozen_chain, pdb_idx_nat) not in nat2hal_pdb_idx:
            print(f'The frozen residue {args.frozen_chain, pdb_idx_nat} is not in the contig. This may or may not be a problem. Just FYI.')
            continue

        # Find equivalent residues in both structures
        pose_idx_nat = p_nat.pdb_info().pdb2pose(args.frozen_chain, pdb_idx_nat)
        pose_idx_hal = p_hal.pdb_info().pdb2pose(*nat2hal_pdb_idx[(args.frozen_chain, pdb_idx_nat)])

        # Force the nat amino acid
        aa_nat = p_nat.residue(pose_idx_nat).name()
        aa_hal = p_hal.residue(pose_idx_hal).name()
        print(f'Forcing p_hal {pose_idx_hal} to {aa_nat}')
        mvr = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(pose_idx_hal, aa_nat)
        mvr.apply(p_hal)
        
        # Mark interface residues so XML can restrict to repacking at "HOTSPOT"
        p_hal.pdb_info().add_reslabel(pose_idx_hal, "HOTSPOT")
        
        # clean the pssm at the pose_idx
        pssm[pose_idx_hal - 1] = 0
            
        # MAY CONSIDER ADDING PAIRWISE CONSTRAINTS FROM HALLUCINATION IN THE FUTURE
        
    # save the modified pssm
    design_utils.save_pssm(pssm, pssm_f_updated)

    # 3. Align p_hal to p_nat by frozen residue bb atoms
    align_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    for pdb_idx_nat in args.freeze_native_residues:
        # Check frozen residue is in the contig. Sometimes we accidently overspecify the frozen residues
        if (args.frozen_chain, pdb_idx_nat) not in nat2hal_pdb_idx:
            print(f'The frozen residue {args.frozen_chain, pdb_idx_nat} is not in the contig. This may or may not be a problem. Just FYI.')
            continue

        # Find equivalent residues in both structures
        pose_idx_nat = p_nat.pdb_info().pdb2pose(args.frozen_chain, pdb_idx_nat)
        pose_idx_hal = p_hal.pdb_info().pdb2pose(*nat2hal_pdb_idx[(args.frozen_chain, pdb_idx_nat)])
      
        for atom in ["N", "CA", "C"]:
            res_hal = p_hal.residue(pose_idx_hal)
            res_nat = p_nat.residue(pose_idx_nat)
            atom_index = res_hal.atom_index(atom)  # this is the same number for either residue
          
            # Add frozen bb atoms to alignment map
            atom_id_nat = pyrosetta.rosetta.core.id.AtomID(atom_index, pose_idx_nat)
            atom_id_hal = pyrosetta.rosetta.core.id.AtomID(atom_index, pose_idx_hal)
            align_map[atom_id_hal] = atom_id_nat
            
    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_hal, p_nat, align_map)

    # 4. Append p_tar to p_hal and clean up the fold tree
    
    # set p_tar to chain B, nummber sequentially
    for i in range(1, p_tar.total_residue()+1):
      p_tar.pdb_info().chain(i, 'B')
      p_tar.pdb_info().number(i, i)
    
    # concatenate hal ptn to the target
    p_hal.append_pose_by_jump(p_tar, 1 )
    p_hal_tar = p_hal
    
    # reverse the fold tree so that the last atom is the root
    ft = p_hal_tar.fold_tree()
    ft.reorder( p_hal_tar.size() ) # reverse the fold tree
    p_hal_tar.fold_tree(ft)
    
    # check things are okay
    print(p_hal_tar.pdb_info())
    print(p_hal_tar.fold_tree())

    
    # 5. Constrain p_hal to stay in place. Must be done after the two poses are combined
    for pdb_idx_nat in args.freeze_native_residues:
        # Check frozen residue is in the contig. Sometimes we accidently overspecify the frozen residues
        if (args.frozen_chain, pdb_idx_nat) not in nat2hal_pdb_idx:
            print(f'The frozen residue {args.frozen_chain, pdb_idx_nat} is not in the contig. This may or may not be a problem. Just FYI.')
            continue
            
        # Find equivalent residues in both structures 
        # (p_hal is still chain A in p_hal_tar, with the original pdb indexing, so this mapping still works)
        pose_idx_nat = p_nat.pdb_info().pdb2pose(args.frozen_chain, pdb_idx_nat)
        pose_idx_hal_tar = p_hal_tar.pdb_info().pdb2pose(*nat2hal_pdb_idx[(args.frozen_chain, pdb_idx_nat)])
        
        # Add coordinate constraints to frozen bb atoms to keep them in place once aligned
        for atom in ["N", "CA", "C"]:
            res_hal_tar = p_hal_tar.residue(pose_idx_hal_tar)
            res_nat = p_nat.residue(pose_idx_nat)
            atom_index = res_hal_tar.atom_index(atom)  # this is the same number for either residue
            tolerance = 0.05  # originally 0.3
            func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(0, tolerance)
            cst = pyrosetta.rosetta.core.scoring.constraints.CoordinateConstraint(
                pyrosetta.rosetta.core.id.AtomID(atom_index, pose_idx_hal_tar),  # atom you want to keep in place
                pyrosetta.rosetta.core.id.AtomID(1, p_hal_tar.size()),  # use root of the fold tree as the "virtual atom"
                res_nat.xyz(atom_index),  # xyz coordinates to keep Atom1 at
                func
            )
            p_hal_tar.add_constraint(cst)
    
    p_hal_tar.dump_pdb(f'{out_dir}/fast_designs/complex/{bn_des}_pre_fd.pdb')
    
    print("Constraints after all mutations and adding target")
    print(p_hal_tar.constraint_set())
    
    # Double check that the coordinate constrants are being scored
    '''
    sfxn = get_fa_scorefxn()
    cc = pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint
    sfxn.set_weight(cc, 1.0)
    apc = pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint
    sfxn.set_weight(apc, 1.0)
    
    sfxn.show(p_hal_tar)
    exit()
    '''
    
    ############################################################################################
    # Run design
    ###########################################################################################
    hbnet_resnums_str = ','.join([str(x) for x in hbnet_resnums])
    
    xml_rd1 = mk_xml(weight_file, min_aa_probability, sequnce_profile_weight, args.layer_design, pssm_f_updated, hbnet_resnums_str, cart=True)  # just a string replacement operation
    task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml_rd1)
    task_relax.setup() # syntax check
    
    # Run fd X number of times until a designs moves < 1.2 A from the initial position
    pre_fd_xyz = get_xyz(p_hal_tar.split_by_chain()[1], atom_id='CA')
    n_attempts = 1
    poses_rmsd = []
    poses_designed = []
    
    for i in range(1, n_attempts+1):
      # check constraint satisfaction before des
      sfxn_cnst = ScoreFunction() 
      cc = pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint
      sfxn_cnst.set_weight(cc, 1.0)
      apc = pyrosetta.rosetta.core.scoring.ScoreType.atom_pair_constraint
      sfxn_cnst.set_weight(apc, 1.0)
      print('Constraint energies before fd')
      clone = p_hal_tar.clone()
      sfxn_cnst(clone)
      print(clone.energies())
      
      print(f"Running protocol. Attempt #{i}")
      packed_pose = task_relax(p_hal_tar)
      p_packed = packed_pose.pose
      poses_designed.append(p_packed)
      
      print('Constraint energies after fd')
      clone = p_packed.clone()
      sfxn_cnst(clone)
      print(clone.energies())

      post_fd_xyz = get_xyz(p_packed.split_by_chain()[1], atom_id='CA')
      diff_xyz = pre_fd_xyz - post_fd_xyz
      rmsd_from_init = np.sqrt((diff_xyz**2).sum(1).mean())
      poses_rmsd.append(rmsd_from_init)
      print(f"Design rmsd_from_init: {rmsd_from_init}")
      
      if rmsd_from_init < 1.2:
        print(f"Made a design with rmsd_from_init < 1.2A!")
        print(f"rmsd for all design attempts: {poses_rmsd}")
        break
        
      if (rmsd_from_init > 1.2) & (i==n_attempts):
        print(f"Failed to find a design with rmsd_from_init < 1.2A with \
              {n_attempts} attempts. Outputting the pose with the smallest rmsd_from_init")
        print(f"rmsd for all design attempts: {poses_rmsd}")
        poses_rmsd = np.array(poses_rmsd)
        p_packed = poses_designed[poses_rmsd.argmin()]

    t1 = time.time()
    print("Design took ", t1-t0)

    # redo pdb_info
    p_packed.pdb_info(p_hal_tar.pdb_info())
    print(p_packed.pdb_info())
    print(p_hal_tar.pdb_info())
    
    df_scores = pandas.DataFrame.from_records([packed_pose.scores])
    
    ###########################################
    # Save results
    ###########################################
    p_packed.dump_pdb(f'{out_dir}/fast_designs/complex/{bn_des}_complex.pdb')
    chA = p_packed.split_by_chain()[1]
    chA.dump_pdb(f'{out_dir}/fast_designs/chA/{bn_des}.pdb')

    #Store the native hbond energies and the corresponding designed hbonds
    df_hbnets_design = hbu.compute_hbond_energies(f'fast_designs/chA/{bn_des}.pdb', from_pose=packed_pose.pose, vsasa_cutoff=0.1, only_return_buried_sc_hbonds=False, exclude_bb=False)
    df_hbnets_design.to_csv(f'{out_dir}/fast_designs/hb_stats/{bn_des}.afterDesign.csv')


    # Iterate over each pair of residues checking that all pairwise hbonds are formed
    # and calculate the hbnet energy between each pair. (note that the dataframes
    # cannot be simply merged on a per hbond basis, due to the symmetry of donor
    # acceptor features // for instance Asp has OD1 and OD2, which are equivalent)
    native_hbond_res_pairs = set(zip(*[source_networks['don_res_n_design'], source_networks['acc_res_n_design'], source_networks['category']]))
    d = {'don_res_n':[], 'acc_res_n':[], 'category': [], 'delta_hbnet_energy':[], 'is_all_hbonds_made':[]}

    n_hbonds_native, n_hbonds_design = 0,0
    for don, acc, cat in native_hbond_res_pairs:
        df_sub_native = source_networks[(source_networks['don_res_n_design']==don) & (source_networks['acc_res_n_design']==acc)]
        df_sub_design = df_hbnets_design[(df_hbnets_design['don_res_n']==don) & (df_hbnets_design['acc_res_n']==acc)]

        n_hbonds_design += len(df_sub_design)
        n_hbonds_native += len(df_sub_native)

        hbonds_energy_native = df_sub_native['energy_hbond_plus_elec'].sum()
        hbonds_energy_design = df_sub_design['energy_hbond_plus_elec'].sum()
        delta = hbonds_energy_design - hbonds_energy_native

        n_hbonds_in_native = len(df_sub_native)
        n_hbonds_in_design = len(df_sub_design)

        d['is_all_hbonds_made'].append(n_hbonds_in_design>=n_hbonds_in_native)
        d['don_res_n'].append(don)
        d['acc_res_n'].append(acc)
        d['category'].append(cat)
        d['delta_hbnet_energy'].append(delta)

    # Summarize the hbnet info
    df_hbnets = pd.DataFrame.from_dict(d)
    mean_hbnet_energy_delta = np.mean(df_hbnets['delta_hbnet_energy'])
    df_hbnets.to_csv(f'{out_dir}/fast_designs/hb_stats/{bn_des}.hbnetAnalysis.csv')

    # Load in the scores add the hbnet scoring summary to that
    df_scores['mean_hbnet_energy_delta'] = mean_hbnet_energy_delta
    df_scores['n_hbonds_native'] = n_hbonds_native
    df_scores['n_hbonds_design'] = n_hbonds_design
    df_scores['name'] = bn_des

    df_scores.to_csv(f'{out_dir}/fast_designs/hb_stats/{bn_des}.rd1.sc')
    print('done')


if __name__ == '__main__':
    main()
