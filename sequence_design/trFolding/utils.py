import numpy as np
import random
from pyrosetta import *

def gen_rst(npz, params):

    dist,omega,theta,phi = npz['dist'],npz['omega'],npz['theta'],npz['phi']

    # dictionary to store Rosetta restraints
    rst = {'dist' : [], 'omega' : [], 'theta' : [], 'phi' : [], 'rep' : []}

    ########################################################
    # assign parameters
    ########################################################
    PCUT  = 0.05 #params['PCUT']
    PCUT1 = params['PCUT1']
    EBASE = params['EBASE']
    EREP  = params['EREP']
    DREP  = params['DREP']
    PREP  = params['PREP']
    SIGD  = params['SIGD']
    SIGM  = params['SIGM']
    MEFF  = params['MEFF']
    DCUT  = params['DCUT']
    ALPHA = params['ALPHA']

    DSTEP = params['DSTEP']
    ASTEP = np.deg2rad(params['ASTEP'])

    seq = params['seq']


    ########################################################
    # dist: 0..20A
    ########################################################
    nres = dist.shape[0]
    bins = np.array([4.25+DSTEP*i for i in range(32)])
    prob = np.sum(dist[:,:,5:], axis=-1)
    bkgr = np.array((bins/DCUT)**ALPHA)
    attr = -np.log((dist[:,:,5:]+MEFF)/(dist[:,:,-1][:,:,None]*bkgr[None,None,:]))+EBASE
    repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(EREP)[None,None,:]
    dist = np.concatenate([repul,attr], axis=-1)
    bins = np.concatenate([DREP,bins])
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    #nbins = 35
    step = 0.5
    for a,b,p in zip(i,j,prob):
        if b>a:
            y = pyrosetta.rosetta.utility.vector1_double()
            _ = [y.append(v) for v in dist[a,b]]
            spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, step, x,y)
            ida = rosetta.core.id.AtomID(5,a+1)
            idb = rosetta.core.id.AtomID(5,b+1)
            rst['dist'].append([a,b,p,rosetta.core.scoring.constraints.AtomPairConstraint(ida, idb, spline)])
    print("dist restraints:  %d"%(len(rst['dist'])))


    ########################################################
    # omega: -pi..pi
    ########################################################
    nbins = omega.shape[2]-1+4
    bins = np.linspace(-np.pi-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    prob = np.sum(omega[:,:,1:], axis=-1)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    omega = -np.log((omega+MEFF)/(omega[:,:,-1]+MEFF)[:,:,None])
    omega = np.concatenate([omega[:,:,-2:],omega[:,:,1:],omega[:,:,1:3]],axis=-1)
    for a,b,p in zip(i,j,prob):
        if b>a:
            y = pyrosetta.rosetta.utility.vector1_double()
            _ = [y.append(v) for v in omega[a,b]]
            spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, ASTEP, x,y)
            id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
            id4 = rosetta.core.id.AtomID(2,b+1) # CA-j
            rst['omega'].append([a,b,p,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])
    print("omega restraints: %d"%(len(rst['omega'])))


    ########################################################
    # theta: -pi..pi
    ########################################################
    prob = np.sum(theta[:,:,1:], axis=-1)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    theta = -np.log((theta+MEFF)/(theta[:,:,-1]+MEFF)[:,:,None])
    theta = np.concatenate([theta[:,:,-2:],theta[:,:,1:],theta[:,:,1:3]],axis=-1)
    for a,b,p in zip(i,j,prob):
        if b!=a:
            y = pyrosetta.rosetta.utility.vector1_double()
            _ = [y.append(v) for v in theta[a,b]]
            spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, ASTEP, x,y)
            id1 = rosetta.core.id.AtomID(1,a+1) #  N-i
            id2 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id3 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id4 = rosetta.core.id.AtomID(5,b+1) # CB-j
            rst['theta'].append([a,b,p,rosetta.core.scoring.constraints.DihedralConstraint(id1,id2,id3,id4, spline)])

    print("theta restraints: %d"%(len(rst['theta'])))


    ########################################################
    # phi: 0..pi
    ########################################################
    nbins = phi.shape[2]-1+4
    bins = np.linspace(-1.5*ASTEP, np.pi+1.5*ASTEP, nbins)
    x = pyrosetta.rosetta.utility.vector1_double()
    _ = [x.append(v) for v in bins]
    prob = np.sum(phi[:,:,1:], axis=-1)
    i,j = np.where(prob>PCUT)
    prob = prob[i,j]
    phi = -np.log((phi+MEFF)/(phi[:,:,-1]+MEFF)[:,:,None])
    phi = np.concatenate([np.flip(phi[:,:,1:3],axis=-1),phi[:,:,1:],np.flip(phi[:,:,-2:],axis=-1)], axis=-1)
    for a,b,p in zip(i,j,prob):
        if b!=a:
            y = pyrosetta.rosetta.utility.vector1_double()
            _ = [y.append(v) for v in phi[a,b]]
            spline = rosetta.core.scoring.func.SplineFunc("", 1.0, 0.0, ASTEP, x,y)
            id1 = rosetta.core.id.AtomID(2,a+1) # CA-i
            id2 = rosetta.core.id.AtomID(5,a+1) # CB-i
            id3 = rosetta.core.id.AtomID(5,b+1) # CB-j
            rst['phi'].append([a,b,p,rosetta.core.scoring.constraints.AngleConstraint(id1,id2,id3, spline)])
    print("phi restraints:   %d"%(len(rst['phi'])))

    return rst

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi,psi=random_dihedral()
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)


#pick phi/psi randomly from:
#-140  153 180 0.135 B
# -72  145 180 0.155 B
#-122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return(phi, psi)


def read_fasta(file):
    fasta=""
    with open(file, "r") as f:
        for line in f:
            if(line[0] == ">"):
                continue
            else:
                line=line.rstrip()
                fasta = fasta + line;
    return fasta


def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


def add_rst(pose, rst, sep1, sep2, params, nogly=False):

    pcut=params['PCUT']
    seq = params['seq']

    # collect restraints
    array = []

    if nogly==True:
        array += [r for a,b,p,r in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut]
        if params['USE_ORIENT'] == True:
            array += [r for a,b,p,r in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut+0.5] #0.5
            array += [r for a,b,p,r in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut+0.5] #0.5
            array += [r for a,b,p,r in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and seq[a]!='G' and seq[b]!='G' and p>=pcut+0.6] #0.6
    else:
        array += [r for a,b,p,r in rst['dist'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut]
        if params['USE_ORIENT'] == True:
            array += [r for a,b,p,r in rst['omega'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut+0.5]
            array += [r for a,b,p,r in rst['theta'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut+0.5]
            array += [r for a,b,p,r in rst['phi'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut+0.6] #0.6

    if len(array) < 1:
        return

    random.shuffle(array)

    cset = rosetta.core.scoring.constraints.ConstraintSet()
    [cset.add_constraint(a) for a in array]

    # add to pose
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_set(cset)
    constraints.add_constraints(True)
    constraints.apply(pose)

