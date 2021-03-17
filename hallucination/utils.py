# load libraries
import numpy as np
import string, sys, getopt, copy, random
from operator import itemgetter

# ivan's natural AA composition
AA_COMP = np.array([0.07892653, 0.04979037, 0.0451488 , 0.0603382 , 0.01261332,
                    0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                    0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                    0.06123779, 0.0547427 , 0.01489194, 0.03705282, 0.0691271])

# STANDARD MODE BIAS - from /home/davidcj/projects/TrR_for_design_v2/design_round2/
TR_AA_FREQ_STAN = np.array([0.01597168, 0.02502841, 0.02988023, 0.10575225, 0.07307457,
                            0.02020281, 0.12834164, 0.05110587, 0.00535012, 0.09485969,
                            0.06157007, 0.09948422, 0.02655827, 0.01981817, 0.15902614,
                            0.01212519, 0.01049917, 0.00893435, 0.00884693, 0.04357024])

AA_REF = np.log(TR_AA_FREQ_STAN/AA_COMP)

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

def parse_PDB(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''
  xyz,seq,min_resn,max_resn = {},{},np.inf,-np.inf
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if chain is None or ch in chain:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): resa,resn = resn[-1],int(resn[:-1])-1
        else: resa,resn = "",int(resn)-1
        
        if resn < min_resn: min_resn = resn
        if resn > max_resn: max_resn = resn
          
        if resn not in xyz: xyz[resn] = {}
        if resa not in xyz[resn]: xyz[resn][resa] = {}
          
        if resn not in seq: seq[resn] = {}
        if resa not in seq[resn]: seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  for resn in range(min_resn,max_resn+1):
    if resn in seq:
      for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
    else: seq_.append(20)
    if resn in xyz:
      for k in sorted(xyz[resn]):
        for atom in atoms:
          if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
          else: xyz_.append(np.full(3,np.nan))
    else:
      for atom in atoms: xyz_.append(np.full(3,np.nan))
  return np.array(xyz_).reshape(-1,len(atoms),3), np.array(seq_)

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

def extend(a,b,c, L,A,D):
  '''
  input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
  output: 4th coord
  '''
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True))
  bc = N(b-c)
  n = N(np.cross(b-a, bc))
  m = [bc,np.cross(n,bc),n]
  d = [L*np.cos(A), L*np.sin(A)*np.cos(D), -L*np.sin(A)*np.sin(D)]
  return c + sum([m*d for m,d in zip(m,d)])

def to_len(a,b):
  '''given coordinates a-b, return length or distance'''
  return np.sqrt(np.sum(np.square(a-b),axis=-1))

def to_len_pw(a,b=None):
  '''given coordinates a-b return pairwise distance matrix'''
  a_norm = np.square(a).sum(-1)
  if b is None: b,b_norm = a,a_norm
  else: b_norm = np.square(b).sum(-1)
  return np.sqrt(np.abs(a_norm.reshape(-1,1) + b_norm - 2*(a@b.T)))

def to_ang(a,b,c):
  '''given coordinates a-b-c, return angle'''
  D = lambda x,y: np.sum(x*y,axis=-1)
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  return np.arccos(D(N(b-a),N(b-c)))

def to_dih(a,b,c,d):
  '''given coordinates a-b-c-d, return dihedral'''
  D = lambda x,y: np.sum(x*y,axis=-1)
  N = lambda x: x/np.sqrt(np.square(x).sum(-1,keepdims=True) + 1e-8)
  bc = N(b-c)
  n1 = np.cross(N(a-b),bc)
  n2 = np.cross(bc,N(c-d))
  return np.arctan2(D(np.cross(n1,bc),n2),D(n1,n2))

def prep_input(pdb, chain=None):
  '''Parse PDB file and return features compatible with TrRosetta'''
  ncac, seq, pdb_idx = parse_PDB_doug(pdb,["N","CA","C"], chain=chain)
  
  # mask gap regions
  mask = seq != 20
  ncac, seq = ncac[mask], seq[mask]

  N,CA,C = ncac[:,0], ncac[:,1], ncac[:,2]
  CB = extend(C, N, CA, 1.522, 1.927, -2.143)

  dist_ref  = to_len(CB[:,None], CB[None,:])
  omega_ref = to_dih(CA[:,None], CB[:,None], CB[None,:], CA[None,:])
  theta_ref = to_dih( N[:,None], CA[:,None], CB[:,None], CB[None,:])
  phi_ref   = to_ang(CA[:,None], CB[:,None], CB[None,:])

  def mtx2bins(x_ref, start, end, nbins, mask):
    bins = np.linspace(start, end, nbins)
    x_true = np.digitize(x_ref, bins).astype(np.uint8)
    x_true[mask] = 0
    return np.eye(nbins)[x_true]

  p_dist  = mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
  p_omega = mtx2bins(omega_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
  p_theta = mtx2bins(theta_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
  p_theta_T = np.transpose(p_theta, [1,0,2])
  p_phi   = mtx2bins(phi_ref,      0.0, np.pi, 13, mask=(p_dist[...,0]==1))
  p_phi_T = np.transpose(p_phi, [1,0,2])
  
  feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega, p_theta_T, p_phi_T],-1)
  return {"seq":N_to_AA(seq), "feat":feat, "dist_ref":dist_ref, "pdb_idx":pdb_idx}

def split_feat(feat):
  out = {}
  for k,i,j in [["theta",0,25],["phi",25,38],["dist",38,75],["omega",75,100]]:
    out[k] = feat[...,i:j]
  return out

def pairwise_id(x):
  '''get pairwise sequence identity'''
  x = np.array(x)
  return (x[:,None] == x[None,:]).mean(-1)

def arr2str(x, d=3):
  return np.array2string(x,formatter={'float_kind':lambda x: f"%.{d}f" % x}).replace("\n","").replace(" ",",")

##################################################################
# functions for parsing masks and converting into moves
##################################################################

def parse_mask(mask):
  '''parse mask option" into [start,end,min,max]'''
  def split_int(x,sep): return [int(y) for y in x.split(sep)]
  out = []
  for x in mask.split(","):
    if ":" in x:
      ran, indel = x.split(":")
      if "-" in indel: min_len, max_len = split_int(indel,"-")
      else: min_len = max_len = int(indel)
    else: ran,indel = x,None
    
    if "-" in ran: start,end = split_int(ran,"-")
    else: start = end = int(ran)
    
    if indel is None: min_len = max_len = (end-start)+1
    
    out.append([start,end,min_len,max_len])
  return out

def mask_to_moves(mask,L):
  '''convert mask into moves (insert/delete)'''
  ok_pdb = np.ones(L,dtype=np.int)
  moves = []
  min_loop_len, max_loop_len = 0,0
  if mask is not None:
    for s,e,mn,mx in parse_mask(mask):
      min_loop_len += mn
      max_loop_len += mx
      
      ln = (e-s)+1
      move = []
      for a in range(mn,mx+1):
        if a > ln: move.append(["i",a,[s]*(a-ln)])
        elif a < ln: move.append(["d",a,(np.arange(ln-a)+s).tolist()])
        else: move.append([None,a,None])
      moves.append(move)
      ok_pdb[s:(e+1)] = 0
  min_len = ok_pdb.sum() + min_loop_len
  max_len = ok_pdb.sum() + max_loop_len
  return moves, ok_pdb, min_len, max_len

def sample_move(moves):
  '''sample move from list of moves'''
  i,d,loop_len = [],[],""
  for move in moves:
    r = np.random.randint(0,len(move),size=1)[0]
    mode,ln,move = move[r]
    loop_len += f"L{ln}"
    if mode == "i":
      for m in move: i.append(m)
    if mode == "d":
      for m in move: d.append(m)
  return [i,d],loop_len

def apply_move(a, move, axes=[0], val=0):
  '''apply [i]nsertion and [d]eletion moves to [a]rray'''
  i_idx, d_idx = move
  idx = np.insert(np.arange(a.shape[axes[0]]),i_idx,-1)
  d_idx = [np.where(idx == d)[0][0] for d in d_idx]
  for i in axes: a = np.insert(a,i_idx,val,axis=i)
  for i in axes: a = np.delete(a,d_idx,axis=i)
  return a

#####################################################################
# Working with multiple sequence alignments
#####################################################################

def parse_fasta(filename, a3m=False):
  '''function to parse fasta file'''
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if len(line) > 0:
      if line[0] == ">":
        header.append(line[1:])
        sequence.append([])
      else:
        if a3m: line = line.translate(rm_lc)
        else: line = line.upper()
        sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return header, sequence

def mk_msa(seqs):
  '''one hot encode msa'''
  alphabet = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alphabet)

  alpha = np.array(alphabet, dtype='|S1').view(np.uint8)
  msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
  for n in range(states):
    msa[msa == alpha[n]] = n
  msa[msa > states] = states-1

  return np.eye(states)[msa]

def get_dist_acc(pred, true, true_mask=None,sep=5,eps=1e-8):
  ## compute accuracy of CB features ##
  pred,true = [x[...,39:51].sum(-1) for x in[pred,true]]
  if true_mask is not None:
    mask = true_mask[:,:,None] * true_mask[:,None,:]
  else: mask = np.ones_like(pred)
  i,j = np.triu_indices(pred.shape[-1],k=sep)
  P,T,M = pred[...,i,j], true[...,i,j], mask[...,i,j]
  ## give equal weighting to positive and negative predictions
  pos = (T*P*M).sum(-1)/((M*T).sum(-1)+eps)
  neg = ((1-T)*(1-P)*M).sum(-1)/((M*(1-T)).sum(-1)+eps)
  return 2.0*(pos*neg)/(pos+neg+eps)

def inv_cov(Y):
  '''given MSA, return contacts'''

  N,L = Y.shape
  K = Y.max()+1
  Y = np.eye(K)[Y]

  # flatten msa (N,L,A) -> (N,L*A)
  Y_flat = Y.reshape(N,-1)

  # compute covariance matrix (L*A,L*A)
  c = np.cov(Y_flat.T)
  # compute shrinkage (l2 regularization)
  shrink = 4.5/np.sqrt(N) * np.eye(c.shape[0])
  # take the inverse to solve for w
  ic = np.linalg.inv(c + shrink)
  # (L,A,L,A)
  ic = ic.reshape(L,K,L,K)

  # take l2norm to reduce (L,A,L,A) to (L,L) matrix
  ic_norm = np.sqrt(np.square(ic).sum((1,3)))
  np.fill_diagonal(ic_norm,0)

  #Average product correction (aka remove largest eigenvector)
  ap = ic_norm.sum(0)
  apc = ic_norm - (ap[:,None]*ap[None,:])/ap.sum()
  np.fill_diagonal(apc,0.0)
  return apc

def to_dict(label, var_list):
  return dict(zip(label,var_list))

def to_list(label, var_dict, default=None):
  return [var_dict.get(k, default) for k in label]

# class for parsing arguments
class parse_args:
  def __init__(self):
    self.long,self.short = [],[]
    self.info,self.help = [],[]

  def txt(self,help):
    self.help.append(["txt",help])

  def add(self, arg, default, type, help=None):
    self.long.append(arg[0])
    key = arg[0].replace("=","")
    self.info.append({"key":key, "type":type,
                      "value":default, "arg":[f"--{key}"]})
    if len(arg) == 2:
      self.short.append(arg[1])
      s_key = arg[1].replace(":","")
      self.info[-1]["arg"].append(f"-{s_key}")
    if help is not None:
      self.help.append(["opt",[arg,help]])

  def parse(self,argv):
    for opt, arg in getopt.getopt(argv,"".join(self.short),self.long)[0]:
      for x in self.info:
        if opt in x["arg"]:
          if x["type"] is None: x["value"] = (x["value"] == False)
          else: x["value"] = x["type"](arg)

    opts = {x["key"]:x["value"] for x in self.info}
    print(str(opts).replace(" ",""))
    return dict2obj(opts)

  def usage(self, err):
    for type,info in self.help:
      if type == "txt": print(info)
      if type == "opt":
        arg, helps = info
        help = helps[0]
        if len(arg) == 1: print("--%-15s : %s"     % (arg[0],help))
        if len(arg) == 2: print("--%-10s -%-3s : %s" % (arg[0],arg[1].replace(":",""),help))
        for help in helps[1:]: print("%19s %s" % ("",help))
    sys.exit(err)

class dict2obj():
  def __init__(self, dictionary):
    for key in dictionary:
      setattr(self, key, dictionary[key])

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
##################################################################################
# Doug's utils
##################################################################################
def force_contig_geo(feat_hal, feat_ref, trb):
  '''
  Force the one-hot geometry of the contigs in the reference pdb
  into the contigs of the hallucinated protein.
  
  Inputs
  -feat_hal [L,L,?] : hallucinated 6D geometries
  -feat_ref [m,m,?] : reference 6D one-hot geometries
  -trb              : tracker dictionary. Must have con_ref_idx0 and con_hal_idx0 fields
  '''
  print('were forcing dem contigs!!!!!!!!!!!!')
  idx0_ref = trb['con_ref_idx0'][0]
  idx0_hal = trb['con_hal_idx0'][0]
  feat_hal[idx0_hal[:,None], idx0_hal[None,:]] = feat_ref[idx0_ref[:,None], idx0_ref[None,:]] + 1e-8
  return feat_hal
  
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

def static_idxs(contigs, hal_len):
  '''
  Randomly place contigs in hallucinted protein, preserving order
  INPUT
  contigs: [[s1,e1], [s2,e2]...]
  hal_len: length of protein to be hallucinated
  
  OUTPUT
  idxs: Indices of contigs in the protein to be hallucinated
        Ex - [2,3,4,9,10,11,12,44,45,46]
  '''
  ncon = len(contigs)
  con_lens = np.array([y - x + 1 for x,y in contigs])
  idx_start = np.random.choice(np.arange(hal_len - con_lens.sum()), ncon, replace=False)
  idx_start.sort()

  cum_con_len = [0] + [con_lens[:i+1].sum() for i in range(ncon-1)]
  idx_start += cum_con_len

  idxs = []
  for s,l in zip(idx_start, con_lens):
    idxs += list(range(s, s+l))
  return np.array(idxs)

def split_probs(p):
  '''
  Split stack of 6D geo probs into individual marginal distribtions
  '''
  out = {
    'theta': p[..., 0:25],
    'phi': p[..., 35:38],
    'dist': p[..., 38:75],
    'omega': p[..., 75:100],
    'theta_T': p[..., 100:125],
    'phi_T': p[..., 125:]
  }
  return out

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

####################################################
# Doug's optimization methods
####################################################
def MCMC_biased(seq_curr, seq_prop, B_choice, L_start, L_range=20, p_indel=0.3):
  '''
  Should only be used with single sequence design and argmax (not sampling)
  '''
  ################################################
  # Function defs
  ################################################
  def mcmc_mut(seq_in, grad, L_start):
    '''
    Biased mutagenasis of a sequence for MCMC run, including indels
    '''
    def insertion(seq):
      L = seq.shape[-2]
      return np.insert(seq, np.random.randint(0,L), np.eye(20)[np.random.randint(0,20)], axis=-2)
    def deletion(seq):
      L = seq.shape[-2]
      return np.delete(seq, np.random.randint(0,L), axis=-2)

    # Decide what type of step to take
    step_type = np.random.choice(['mutation', 'indel'] , p=[1-p_indel, p_indel])
    if step_type == 'indel':
      L_curr = seq_in.shape[-2]
      if L_curr == L_start+L_range:
        seq_out = deletion(seq_in)
        print('Proposing a deletion...')
      elif L_curr == L_start-L_range:
        seq_out = insertion(seq_in)
        print('Proposing an insertion...')
      elif np.random.rand() < 0.5:
        seq_out = deletion(seq_in)
        print('Proposing a deletion...')
      else:
        seq_out = insertion(seq_in)
        print('Proposing an insertion...')
      prob = None

    if step_type == 'mutation':
      print('Proposing a mutation...')
      # 1. Convert grad to probability
      B_prob = 0  # 50  # 0 is "classic" MCMC
      prob = np.exp(-B_prob * grad)
      prob *= (1 - seq_in)  # exclude current sequence
      prob /= prob.sum()

      # 2. Biased selection of mutation
      idx_1d = np.random.choice(np.arange(prob.size), p=prob.reshape(-1))
      idx_batch, idx_seq, idx_L, idx_aa = np.unravel_index(idx_1d, prob.shape)

      # 3. Update next sequence to evaluate
      seq_out = np.copy(seq_in)  # not entirely sure why did is necesary.
      seq_out[idx_batch, idx_seq, idx_L] = np.eye(20)[idx_aa]

    return seq_out, prob  

  ################################################
  # Main
  ################################################
  # Metropolis criterion to accept/reject new sequence
  delta_loss = seq_prop['loss'] - seq_curr['loss']
  print(f'Delta loss: {delta_loss:.3f}')
  print(f'Chance of acceptance: {min(1, np.exp(-delta_loss * B_choice)):.3f}')
  if np.exp(-delta_loss * B_choice) > np.random.rand():  # accept new sequence
    print('The proposed sequence was accepted!')
    seq_curr = copy.copy(seq_prop)
  else:
    print('The proposed sequence was rejected :(')

  # Make a new mutation
  seq_new, prob = mcmc_mut(seq_curr['seq'], seq_curr['grad'], L_start)

  return seq_new, seq_curr


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