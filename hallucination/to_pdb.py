import numpy as np
from sklearn.manifold import MDS
from utils import *

def metric_MDS(x):
  return MDS(3,dissimilarity="precomputed").fit_transform(x)

def fix_chirality(x):
  n,ca,c = x.reshape(-1,3,3).transpose(1,0,2)
  phi_mu = to_dih(c[:-1],n[1:],ca[1:],c[1:]).mean()
  if phi_mu > 0: return x*[1,1,-1]
  else: return x

def vals_to_xyz(theta, phi, dist, omega, mask=None, verbose=False):
  include=["n","a","c"]
  at = len(include)
  L = dist.shape[0]

  def flat(x):
    np.fill_diagonal(x,np.nan)
    return x.flatten()[:,None]

  d = {"na":1.458,"an":1.458,
       "ac":1.523,"ca":1.523,
       "ab":1.522,"ba":1.522,
       "c_n":1.329,
       "nb":2.447,"bn":2.447,
       "cb":2.499,"bc":2.499,
       "nc":2.460,"cn":2.460,
       "anc":0.615,"banc":-2.143,
       "nab":1.927,"ban":1.927,
       "nabb":flat(theta),"bban":flat(theta.T),
       "abb":flat(phi),"bba":flat(phi.T),
       "bb":flat(dist),"abba":flat(omega)}

  # reconstruct pairwise coordinates
  x,y = {},{}
  x["n"] = np.array([[0,0,0]])
  x["a"] = np.array([[0,0,d["na"]]])
  x["b"] = np.array([[0,d["ab"]*np.sin(d["nab"]),d["na"]-d["ab"]*np.cos(d["nab"])]])
  x["c"] = extend(x["b"],x["a"],x["n"],d["nc"],d["anc"],d["banc"])
  y["b"] = extend(x["n"],x["a"],x["b"],d["bb"],d["abb"],d["nabb"])
  y["a"] = extend(x["a"],x["b"],y["b"],d["ba"],d["bba"],d["abba"])
  y["n"] = extend(x["b"],y["b"],y["a"],d["an"],d["ban"],d["bban"])
  y["c"] = extend(y["b"],y["a"],y["n"],d["nc"],d["anc"],d["banc"])

  # get backbone phi/psi/omega
  yn = y["n"][1::(L+1)]
  ya = y["a"][1::(L+1)]
  yc = y["c"][1::(L+1)]
  psi = to_dih(x["n"],x["a"],x["c"],yn)
  omg = to_dih(x["a"],x["c"],yn,ya)
  phi = to_dih(x["c"],yn,ya,yc)

  # compute distance between coordinates
  dm = []
  for i in (include):
    for j in (include):
      dm_ = to_len(x[i],y[j]).reshape(L,L)
      if i == j: np.fill_diagonal(dm_,0)
      else: np.fill_diagonal(dm_,d[i+j])
      dm.append(dm_)

  sparse_dm = np.array(dm).reshape(at,at,L,L).transpose(2,0,3,1).reshape(at*L,at*L)
  if mask is not None:
    mask = np.tile(mask[:,None,:,None],[1,at,1,at]).reshape(at*L,at*L)
    sparse_dm[mask == 0] = np.inf

  # fix bond lengths
  def fill_diag(x,k,d):
    np.fill_diagonal(x[k:,:],d)
    np.fill_diagonal(x[:,k:],d)

  bonds = [d["na"],d["ac"],d["c_n"]]
  fill_diag(sparse_dm,1,bonds)

  # fill-in missing distances
  full_dm = floyd_warshall(sparse_dm)
  fill_diag(full_dm,1,bonds)

  # recover coordinates from dihedrals/distances
  fL,fA,fD = ppo_to_xlad(psi,omg,phi)
  xyz = fix_chirality(metric_MDS(full_dm))
  xyz = linear_recon(full_dm, X=xyz, L=fL, A=fA, D=fD,
                     W=mask, verbose=verbose, refine_iter=100)
  #xyz = refine(full_dm, xyz, W=mask, verbose=verbose)

  # refine atoms
  N,CA,C = xyz.reshape(L,-1,3).transpose(1,0,2)

  # place CB and O atoms
  CB = extend(C,N,CA,1.522,1.927,-2.143)
  O = extend(np.roll(N,-1,axis=0),CA,C,1.231,2.108,-3.142)
  CA_dm = full_dm[1::3,:][:,1::3]
  return np.stack([N,CA,C,O,CB],1), CA_dm

def bins_to_vals(theta, phi, dist, omega):

  def get_bins(s,e,b):
    a = np.linspace(s,e,b+1)
    return (a[1:]+a[:-1])/2

  def bin2val(a, b, dih=False, eps=1e-8):
    a_sub = a[...,1:]
    if dih: return np.arctan2(bin2val(a,np.sin(b)),bin2val(a,np.cos(b)))
    else: return (a_sub*b).sum(-1)/(a_sub.sum(-1)+eps)

  # get mask
  mask = dist[...,1:].max(-1)
  mask[dist.argmax(-1) == 0] = 0

  # convert bins to values
  len_bins = get_bins(2,20,36)
  ang_bins = get_bins(0,np.pi,12)
  dih_bins = get_bins(-np.pi,np.pi,24)

  dist = bin2val(dist,len_bins)
  phi = bin2val(phi,ang_bins)
  omega = bin2val(omega,dih_bins,dih=True)
  theta = bin2val(theta,dih_bins,dih=True)

  return theta, phi, dist, omega, mask

def linear_recon(DM, X=None, L=None, A=None, D=None, W=None,
                 bond_iter=3, refine_iter=10,
                 verbose=False, DM_native=None, rev=True):

  def np_norm(x, axis=None, keepdims=True, eps=1e-8):
    return np.sqrt(np.sum(np.square(x), axis=axis, keepdims=keepdims) + eps)

  def do_step(x, xn, dn, w, step, bond):
    if step is None: step = np.random.normal(scale=0.01,size=(3,))
    # optimize step
    for _ in range(bond_iter):
      new_x = (x - step)
      # compute gradient
      dn_i = dn[:,None]
      dn_o = np_norm(xn-new_x,-1)
      g = (2 * (new_x - xn) * (dn_o - dn_i))/dn_o
      # weighted gradient
      if w is None: g = g.mean(0)
      else: g = np.sum(g * w[:,None], axis=0)/(np.sum(w[:,None], axis=0) + 1e-8)
      # constrain step to bond-length
      step = g + step
      step = step/np_norm(step) * bond
    return x - step

  def build(fX, fDM, fL, fA=None, fD=None, fW=None, ini=True, rev=False):
    X,DM,L,A,D,W = fX,fDM,fL,fA,fD,fW
    if rev: # flip direction!
      X,L,DM = fX[::-1],fL[::-1],fDM[::-1,::-1]
      if fA is not None: A = fA[::-1]
      if fD is not None: D = fD[::-1]
      if fW is not None: W = fW[::-1,::-1]

    if A is None or D is None: # if angle or dihedral not provided
      for k in range(1,ln):
        if ini: X[k] = do_step(x=X[k-1],xn=X[:k],dn=DM[k,:k],w=None,step=None,bond=L[k-1])
        else: X[k] = do_step(x=X[k-1],xn=X,dn=DM[k],w=W[k],step=X[k-1]-X[k],bond=L[k-1])
    else:
      for k in range(3,ln):
        step = X[k-1] - extend(X[k-3],X[k-2],X[k-1],L[k-1],A[k-2],D[k-3])
        if ini: X[k] = do_step(x=X[k-1],xn=X[:k],dn=DM[k,:k],w=None,step=step,bond=L[k-1])
        else: X[k] = do_step(x=X[k-1],xn=X,dn=DM[k],w=W[k],step=step,bond=L[k-1])

  def rms():
    DM_ = to_len_pw(X)
    return np.sqrt((W*np.square(DM_native-DM_)).sum()/W.sum())

  # initialize
  ln = DM.shape[0]
  if DM_native is None: DM_native = DM
  if W is None: W = np.ones_like(DM)
  if L is None: L = DM.flat[1::(L+1)]
  if X is None:
    X = np.zeros([ln,3])
    # if angle and dihedral are provided
    if A is not None and D is not None:
      X[:3] = [[0,0,0],[0,0,L[0]],
               [0,L[1]*np.sin(A[0]),L[0]-L[1]*np.cos(A[0])]]

    # initial extension
    build(X,DM,L,A,D)
    if verbose: print("ini",rms())

  # refine extension
  for _ in range(refine_iter):
    if rev and _ % 2 == 0: build(X,DM,L,A,D,W,ini=False,rev=True)
    else: build(X,DM,L,A,D,W,ini=False)
    if verbose: print(_,rms())
  return X

def ppo_to_xlad(PSI,OMG,PHI):
  d = {"na":1.458,"ac":1.523,"cn":1.329,
       "nac":1.941,"acn":2.028,"cna":2.124}
  # initialize N,CA,C
  ln = PSI.shape[0]
  L = np.tile([d["cn"],d["na"],d["ac"]],ln)
  L = np.append([d["na"],d["ac"]],L)
  A = np.tile([d["acn"],d["cna"],d["nac"]],ln)
  A = np.append([d["nac"]],A)
  D = np.array([PSI,OMG,PHI]).T.flatten()
  return L,A,D

def refine(dm, xyz, W=None, iter=100, verbose=False):
  if W is None: W = np.ones_like(dm)
  rms = lambda: np.sqrt((W*np.square(dm-to_len_pw(xyz))).sum()/W.sum())
  for p in range(iter):
    # compute gradient
    x_x = (xyz[:,None] - xyz[None,:]).T
    dm_ = np.sqrt(np.square(x_x).sum(0)+1e-8)
    g = (2.0 * x_x * (dm - dm_))/dm_
    g = ((g*W).sum(-1)/W.sum(-1)).T
    # apply gradient
    xyz = xyz - g
    if verbose: print(p,rms())
  return xyz

def floyd_warshall(d):
  for m in range(d.shape[0]):
    o = d[m]
    d = np.minimum(d, o[None,:]+o[:,None])
  return d

def classic_MDS(M,dims=3):
  L = M.shape[0]
  d = np.square(M)
  c = np.eye(L) - np.ones((L,L))/L
  m = -0.5 * c @ d @ c
  s,u = np.linalg.eigh(m)
  X = u[:,-dims:] * np.sqrt(s[-dims:])
  return X

def feat_to_xyz(feat):
  D = split_feat(feat)
  return vals_to_xyz(*bins_to_vals(D["theta"],D["phi"],D["dist"],D["omega"]))

def save_PDB(pdb_out, coords, dm, seq):
  atoms = ['N','CA','C','O','CB']
  error = np.sqrt(np.square(dm-to_len_pw(coords[:,1])).mean(-1))
  out = open(pdb_out,"w")
  k = 0
  for r,residue in enumerate(coords):
    AA = aa_1_3[seq[r]]
    for a,atom in enumerate(residue):
      if AA == "GLY" and atoms[a] == "CB": continue
      x,y,z = atom
      out.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                % (k+1,atoms[a],AA,"A",r+1,x,y,z,1,error[r]))
      k += 1
  out.close()
