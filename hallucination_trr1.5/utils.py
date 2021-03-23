import numpy as np
import scipy
from scipy.special import softmax
import scipy.spatial
import re
from operator import itemgetter

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
  # ["ARND"] -> [[0,1,2,3]]
  x = np.array(x);
  if x.ndim == 0: x = x[None]
  return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

def parse_contigs(contig_input, pdb_id):                                                                                           
  '''
  Input: contig start/end by pdb chain and residue number as in the pdb file
         ex - B12-17
  Output: corresponding start/end indices of the "features" numpy array (idx0)
  '''
  contigs = []
  for con in contig_input.split(','):
    pdb_ch = con[0]
    pdb_s, pdb_e = map(int, con[1:].split('-'))

    np_s = pdb_id.index((pdb_ch, pdb_s))
    np_e = pdb_id.index((pdb_ch, pdb_e))

    contigs.append([np_s, np_e])
  return contigs

def parse_PDB_doug(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence, pdb_idx
  '''
  xyz,seq, pdb_idx = {},{}, []
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if chain is None or ch in chain:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]              # AA identity
        resn = line[22:22+5].strip()      # residue number
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        idx_pdb = (ch, int(resn))
        pdb_idx.append(idx_pdb)
        if idx_pdb not in seq: seq[idx_pdb] = resi
        if idx_pdb not in xyz: xyz[idx_pdb] = {}
        if atom not in xyz[idx_pdb]:
          xyz[idx_pdb][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_,pdb_idx_ = [],[],[]
  for idx_pdb in sorted(seq.keys(), key=itemgetter(0,1)):  # sort by chain, then by idx
    pdb_idx_.append(idx_pdb)
    seq_.append(aa_3_N.get(seq[idx_pdb],20))  # append aa int to seq_
    for atom in atoms:
      if atom in xyz[idx_pdb]: xyz_.append(xyz[idx_pdb][atom])  # append xyz coordinates, one atom at a time
      else: xyz_.append(np.full(3,np.nan))

  return np.array(xyz_).reshape(-1,len(atoms),3), np.array(seq_), pdb_idx_

# calculate dihedral angles defined by 4 sets of points
def get_dihedrals(a, b, c, d):

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

# calculate planar angles defined by 3 sets of points
def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)

# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, dmax):

    nres = xyz.shape[0]

    # three anchor atoms
    N  = xyz[:,0,:]
    Ca = xyz[:,1,:]
    C  = xyz[:,2,:]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres),999.9)
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist6d, omega6d, theta6d, phi6d

def prep_input(pdb, chain=None, double_asym_feat=True):
    '''Parse PDB file and return features compatible with TrRosetta'''
    ncac, seq, pdb_idx = parse_PDB_doug(pdb,["N","CA","C"], chain=chain)

    # mask gap regions
    mask = seq != 20
    ncac, seq = ncac[mask], seq[mask]
    d,o,t,p = get_coords6d(ncac, 20.0)

    # bins
    dstep = 0.5
    astep = np.deg2rad(10.0)
    dbins = np.linspace(2.0+dstep, 20.0, 36)
    ab180 = np.linspace(0.0+astep, np.pi, 18)
    ab360 = np.linspace(-np.pi+astep, np.pi, 36)

    # bin coordinates
    db = np.digitize(d, dbins).astype(np.uint8)
    ob = np.digitize(o, ab360).astype(np.uint8)
    tb = np.digitize(t, ab360).astype(np.uint8)
    pb = np.digitize(p, ab180).astype(np.uint8)

    # synchronize 'no contact' bins
    ob[db == 36] = 36
    tb[db == 36] = 36
    pb[db == 36] = 18

    # 1-hot encode and stack all coords together
    feat = [np.eye(37)[db],
            np.eye(37)[ob],
            np.eye(37)[tb],
            np.eye(19)[pb]]
    print(tb.shape)
    if double_asym_feat:
        feat.append(np.eye(37)[np.transpose(tb,[1,0])])
        feat.append(np.eye(19)[np.transpose(pb,[1,0])])
    feat = np.concatenate(feat, axis=-1)

    return {"seq":N_to_AA(seq), "feat":feat, "dist_ref":d, "pdb_idx":pdb_idx}

# characters to integers
def aa2idx(seq):

    # convert letters into numbers
    abc = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    idx = np.array(list(seq), dtype='|S1').view(np.uint8)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i

    # treat all unknown characters as gaps
    idx[idx > 20] = 20

    return idx

# integers back to sequence
def idx2aa(idx):
    abc=np.array(list("ARNDCQEGHILKMFPSTWYV-"))
    return("".join(list(abc[idx])))

# pssm to 1-hot sequence
def argmax(logit):
    '''converting PSSM to 1-hot sequence'''
    NSEQ,NRES = logit.shape[:-1]
    pssm = softmax(logit,axis=-1)
    #x_inp = np.eye(20)[logit[...,:-1].argmax(-1)]
    x_inp = np.eye(20)[pssm[...,:20].argmax(-1)]
    return x_inp.reshape((NSEQ,NRES,20))

def sample(logit, seq_hard=True):
    NSEQ,NRES,NS = logit.shape
    eps=1e-20
    U = np.random.uniform(size=logit.shape)
    pssm_sampled = softmax(logit-np.log(-np.log(U+eps)+eps),axis=-1)
    if seq_hard:
        x_inp = np.eye(20)[pssm_sampled[...,:20].argmax(-1)]
    else:
        x_inp = pssm_sampled
    return x_inp.reshape((NSEQ,NRES,20))

def cons2idxs(contigs):
  '''
  Convert from [[s1,e1], [s2,e2],...] to 
  [s1,...,e1,s2,...,e2,...]
  '''
  idxs = []
  for s,e in contigs:
    idxs += list(range(s,e+1))
  return idxs

def idxs2cons(idxs):
  '''
  Converts from [1,2,3,4,7,8,9]
  to [[2,4],[7,9]]
  '''
  idxs = np.array(idxs)
  cons = [[idxs[0]]]
  for con_end in np.where(np.diff(idxs) != 1)[0]:
    cons[-1].append(idxs[con_end])
    cons.append([idxs[con_end + 1]])
  cons[-1].append(idxs[-1])
  return cons

# helper function for argparse. allows arguments to be set explicitly as t/f
def str2bool(v):                                                                                       
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def mk_feat_hal_and_mappings(hal_2_ref_idx0, pdb_out):
  #####################################
  # rearrange ref features according to hal_2_ref_idx0
  #####################################
  #1. find corresponding idx0 in hal and ref
  hal_idx0 = []
  ref_idx0 = []
  
  for hal, ref in enumerate(hal_2_ref_idx0):
    if ref is not None:
      hal_idx0.append(hal)
      ref_idx0.append(ref)
      
  hal_idx0 = np.array(hal_idx0, dtype=int)
  ref_idx0 = np.array(ref_idx0, dtype=int)
      
  #2. rearrange the 6D features
  hal_len = len(hal_2_ref_idx0)
  d_feat = pdb_out['feat'].shape[-1]
  
  feat_hal = np.zeros([1, hal_len, hal_len, d_feat])
  feat_ref = pdb_out['feat'][None]
  feat_hal[:, hal_idx0[:,None], hal_idx0[None,:]] = feat_ref[:, ref_idx0[:,None], ref_idx0[None,:]]
      
  #3. make the 1d binary mask, for backwards compatibility
  hal_2_ref_idx0 = np.array(hal_2_ref_idx0, dtype=np.float32)  # convert None to NaN
  mask_1d = (~np.isnan(hal_2_ref_idx0)).astype(float)
  mask_1d = mask_1d[None]
  
  
  #####################################
  # mappings between hal and ref
  #####################################
  mappings = {
    'con_hal_idx0': hal_idx0.tolist(),
    'con_ref_idx0': ref_idx0.tolist(),
    'con_hal_pdb_idx': [('A',i+1) for i in hal_idx0],
    'con_ref_pdb_idx': [pdb_out['pdb_idx'][i] for i in ref_idx0],
    'mask_1d': mask_1d,
  }
  
  return feat_hal, mappings

def apply_mask(mask, pdb_out):
  '''
  Uniformly samples gap lengths, then gathers the ref features
  into the target hal features
  
  Inputs:
  mask: specify the order and ranges of contigs and gaps
        Contig - A continuous range of residues from the pdb.
                Inclusive of the begining and end
                Must start with the chain number
                ex: B6-11
        Gap - a range of gaps lengths the model is free to hallucinate
                Gap ranges are inclusive of the end
                ex: 9-21

        ex - '3-3,B6-11,9-21,A36-42,20-30,A12-24,3-6'
  
  pdb_out: dictionary from the prep_input function
  
  '''
  
  ref_pdb_2_idx0 = {pdb_idx:i for i, pdb_idx in enumerate(pdb_out['pdb_idx'])}
  
  #1. make a map from hal_idx0 to ref_idx0. Has None for gap regions
  hal_2_ref_idx0 = []
  for el in mask.split(','):

    if el[0].isalpha():  # el is a contig
      chain = el[0]
      s,e = el[1:].split('-')
      s,e = int(s), int(e)
      
      for i in range(s, e+1):
        idx0 = ref_pdb_2_idx0[(chain, i)]
        hal_2_ref_idx0.append(idx0)
        
    else:  # el is a gap
      # sample gap length
      s,e = el.split('-')
      s,e = int(s), int(e)
      gap_len = np.random.randint(s, e+1)
      hal_2_ref_idx0 += [None]*gap_len
      
  #2. Convert mask to feat_hal and mappings 
  feat_hal, mappings = mk_feat_hal_and_mappings(hal_2_ref_idx0, pdb_out)
  
  return feat_hal, mappings


def scatter_contigs(contigs, pdb_out, L_range, keep_order=False, min_gap=0):
  '''
  Randomly places contigs in a protein within the length range.
  
  Inputs
    Contig: A continuous range of residues from the pdb.
            Inclusive of the begining and end
            Must start with the chain number
            ex: B6-11
    pdb_out: dictionary from the prep_input function
    L_range: String range of possible lengths.
              ex: 90-110
              ex: 70
    keep_order: keep contigs in the provided order or randomly permute
    min_gap: minimum number of amino acids separating contigs
    
  Outputs
    feat_hal: target pdb features to hallucinate
    mappings: dictionary of ways to convert from the hallucinated protein
              to the reference protein  
  
  '''
  
  ref_pdb_2_idx0 = {pdb_idx:i for i, pdb_idx in enumerate(pdb_out['pdb_idx'])}
  
  #####################################
  # make a map from hal_idx0 to ref_idx0. Has None for gap regions
  #####################################
  #1. Permute contig order
  contigs = contigs.split(',')
  
  if not keep_order:
    random.shuffle(contigs)
    
  #2. convert to ref_idx0
  contigs_ref_idx0 = []
  for con in contigs:
    chain = con[0]
    s, e = map(int, con[1:].split('-'))
    contigs_ref_idx0.append( [ref_pdb_2_idx0[(chain, i)] for i in range(s, e+1)] )
  
  #3. Add minimum gap size
  for i in range(len(contigs_ref_idx0) - 1):
    contigs_ref_idx0[i] += [None] * min_gap
    
  #4. Sample protein length
  if '-' in L_range:
    L_low, L_high = map(int, L_range.split('-'))
  else:
    L_low, L_high = int(L_range), int(L_range)
    
  L_hal = np.random.randint(L_low, L_high+1)
  
  L_con = 0
  for con in contigs_ref_idx0:
    L_con += len(con)
    
  L_gaps = L_hal - L_con
  
  if L_gaps <= 1:
    print("Error: The protein isn't long enough to incorporate all the contigs."
          "Consider reduce the min_gap or increasing L_range")
    return
  
  #5. Randomly insert contigs into gaps
  hal_2_ref_idx0 = np.array([None] * L_gaps, dtype=float)  # inserting contigs into this
  n_contigs = len(contigs_ref_idx0)  
  insertion_idxs = np.random.randint(L_gaps + 1, size=n_contigs)
  insertion_idxs.sort()
  
  for idx, con in zip(insertion_idxs[::-1], contigs_ref_idx0[::-1]):
    hal_2_ref_idx0 = np.insert(hal_2_ref_idx0, idx, con)
    
  #6. Convert mask to feat_hal and mappings
  hal_2_ref_idx0 = [int(el) if ~np.isnan(el) else None for el in hal_2_ref_idx0]  # convert nan to None
  feat_hal, mappings = mk_feat_hal_and_mappings(hal_2_ref_idx0, pdb_out)
    
  return feat_hal, mappings           
