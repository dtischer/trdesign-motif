import numpy as np
import scipy
import scipy.spatial

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

    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[0]
    Ca = xyz[1]
    C  = xyz[2]

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


def get_binned6d(xyz):

    d,o,t,p = get_coords6d(xyz, 20.0)

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
    c6d = np.concatenate([np.eye(37)[db],
                          np.eye(37)[ob],
                          np.eye(37)[tb],
                          np.eye(19)[pb]], axis=-1)

    return c6d


# project 3d points onto local frames
def parse_pocket(filename):

    lines = open(filename,'r').readlines()

    # parse N,Ca,C of pocket residues
    N  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="N"])
    Ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])
    C  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="C"])

    c6d = get_binned6d(np.stack([N,Ca,C]))

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # parse alpha spheres
    points = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                       for l in lines if l[:4]=="HETA" and l[17:20].strip()=="STP"])

    nres = N.shape[0]
    nsph = points.shape[0]

    # spherical coords of a-spheres in
    # local frames of bs residues
    N  = np.repeat(N,nsph,axis=0)
    Ca = np.repeat(Ca,nsph,axis=0)
    Cb = np.repeat(Cb,nsph,axis=0)

    # load 3D grids into kd trees
    kd = []
    for f in ['00','01','02']:
        grids3d = np.loadtxt('/home/aivan/for/GyuRie/trRosetta2/grids3d/grids1023.'+f+'.txt')
        dmin = np.min(np.linalg.norm(grids3d[:,None,:]-grids3d[None,:,:],axis=-1)[np.triu_indices(grids3d.shape[0],1)])
        rmax = np.max(np.linalg.norm(grids3d,axis=-1))
        grids3d /= rmax+dmin/2
        kd.append(scipy.spatial.cKDTree(grids3d*12.0))


    cav = np.zeros((3,nres,1024),dtype=np.uint8)
    for i in range(3):
        for j in range(nres):
            d2s = np.linalg.norm(Cb[j*nsph:(j+1)*nsph]-points, axis=-1)
            t2s = get_dihedrals(N[j*nsph:(j+1)*nsph], Ca[j*nsph:(j+1)*nsph], Cb[j*nsph:(j+1)*nsph], points)
            p2s = get_angles(Ca[j*nsph:(j+1)*nsph], Cb[j*nsph:(j+1)*nsph], points)
            xyz3d = np.stack([
                d2s * np.sin(p2s) * np.cos(t2s),
                d2s * np.sin(p2s) * np.sin(t2s),
                d2s * np.cos(p2s) ]).T
            grids1hot = np.array([np.sum(np.eye(1024)[idx],axis=0)
                                    if len(idx)>0 else np.zeros((1024))
                                    for idx in kd[i].query_ball_point(xyz3d, 2.0)])   # !!! set radius here !!!
            cav[i,j] = np.sum(grids1hot,axis=0)>0


    '''
    points = np.vstack([points]*nres)
    d2s = np.linalg.norm(Cb-points, axis=-1)
    t2s = get_dihedrals(N, Ca, Cb, points)
    p2s = get_angles(Ca, Cb, points)
    print(d2s.shape,t2s.shape,p2s.shape,points.shape)
    xyz3d = np.stack([
        d2s * np.sin(p2s) * np.cos(t2s),
        d2s * np.sin(p2s) * np.sin(t2s),
        d2s * np.cos(p2s) ]).T
                                
    cav2 = np.zeros((3,nres,1024),dtype=np.uint8)
    for i in range(3):
        grids1hot = np.array([np.sum(np.eye(1024)[idx],axis=0)
                                if len(idx)>0 else np.zeros((1024)) 
                                for idx in kd[i].query_ball_point(xyz3d, 2.0)])
        #print(len(kd.query_ball_point(xyz3d, 2.0)),kd.query_ball_point(xyz3d, 2.0))
        cav2[i] = np.sum(grids1hot.reshape([nsph,nres,1024]),axis=0)>0
    '''

    return c6d,cav,points


