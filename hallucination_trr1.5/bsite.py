import numpy as np
from scipy import signal
import tensorflow as tf

def probe_bsite_tf(pred, pred3d, wcav, bsite, cav, beta, L, n):

    '''
    -------------------------------------------------------------------------
    - pred[?,L,L,130]    : trRosetta predictions (tf.tensor)
    - pred3d[3,L,L,1024] : ???
    - wcav               : weight for the cavity term (float)
    - bsite[n,n,130]     : 1-hot encoded binding site, 6D coords (np.array) 
    - cav[3,n,1024]      : pocket volume
    - beta               : inverse temperature, a scalar (float)
    - L                  : protein length (int)
    - n                  : binding site size (int)
    -------------------------------------------------------------------------
    '''

    # extend arrays so that each pixel stores all 6D coordinates, not just 4 of them
    pred = tf.concat([pred,tf.transpose(pred[:,:,:,74:],[0,2,1,3])], axis=-1)
    bsite_tf = tf.constant(bsite,dtype=tf.float32)
    bs_c6d  = tf.concat([bsite,tf.transpose(bsite_tf[:,:,74:],[1,0,2])], axis=-1)

    hash_ = lambda t : hash(np.sort(t).tostring())

    pairs   = [[i,j] for i in range(n-1) for j in range(i+1,n)]
    triples = [[i,j,k] for i in range(n-2) for j in range(i+1,n-1) for k in range(j+1,n)]

    # clash scores
    cav_np = cav + 1e-8
    cav_np /= np.sum(cav_np,axis=-1,keepdims=True)
    cav_tf = tf.constant(cav_np,dtype=tf.float32)

    CCE_i_cav = []
    for i in range(n):
        cav_tf_i = cav_tf[:,i,:]
        #cce_i = tf.reduce_sum(tf.log(pred3d+1e-8)*cav_tf_i[:,None,None,:],axis=[0,2,3])/3/1024
        cce_i = tf.reduce_sum(tf.log(pred3d+1e-8)*cav_tf_i[:,None,None,:],axis=-1)
        cce_i = tf.reduce_mean(cce_i,axis=[0,2])
        CCE_i_cav.append(cce_i)
    CCE_i_cav = tf.stack(CCE_i_cav)

    # satisfaction score + cavity score
    CCE_ij  = {hash_(p): -tf.reduce_sum(bs_c6d[p[0],p[1]][None,None,None,:]*tf.math.log(pred), axis=-1)/6 for p in pairs}
    CCE_ijk = {hash_(t): CCE_ij[hash_([t[0],t[1]])][:,:,:,None]/3 + \
                         CCE_ij[hash_([t[0],t[2]])][:,:,None,:]/3 + \
                         CCE_ij[hash_([t[1],t[2]])][:,None,:,:]/3 + \
                         wcav*CCE_i_cav[None,t[0],:,None,None]/3 + \
                         wcav*CCE_i_cav[None,t[1],None,:,None]/3 + \
                         wcav*CCE_i_cav[None,t[2],None,None,:]/3 for t in triples}


    CCE_exp_ijk = {k: tf.math.exp(-tf.cast(beta*cce,dtype=tf.float64)) for k,cce in CCE_ijk.items()}
    P_ijk   = {k: tf.cast(CCE_exp_ijk[k]/tf.reduce_sum(CCE_exp_ijk[k],axis=(1,2,3),keepdims=True),dtype=tf.float32) for k,cce in CCE_ijk.items()}
    cce_sat = tf.reduce_mean([tf.reduce_sum(cce*P_ijk[k],axis=(1,2,3)) for k,cce in CCE_ijk.items()],axis=0)

    #P_ijk   = {k: tf.math.exp(-tf.cast(beta*cce,dtype=tf.float64))/tf.reduce_sum(tf.math.exp(-tf.cast(beta*cce,dtype=tf.float64)),axis=(1,2,3))[:,None,None,None] for k,cce in CCE_ijk.items()}
    #cce_sat = tf.reduce_mean([tf.reduce_sum(cce*tf.cast(P_ijk[k],dtype=tf.float32),axis=(1,2,3)) for k,cce in CCE_ijk.items()],axis=0) 
            

    # consistency score
    cce_consist = []
    for p in np.array(pairs):
        diff = np.setdiff1d(np.arange(n),p)
        for i in range(n-3):
            pi = tf.reduce_sum(P_ijk[hash_([p[0],p[1],diff[i]])], axis=np.sum(p<diff[i])+1)
            for j in range(i+1,n-2):
                pj = tf.reduce_sum(P_ijk[hash_([p[0],p[1],diff[j]])], axis=np.sum(p<diff[j])+1)
                cce = 0.5*(-tf.reduce_sum(pi*tf.math.log(pj+1e-8),axis=(1,2))-tf.reduce_sum(pj*tf.math.log(pj+1e-8),axis=(1,2)))
                cce_consist.append(cce)
    cce_consist = tf.reduce_mean(cce_consist, axis=0)

    # 3-site probabilities projected down to 2d and
    # averaged over all projections and triples
    p2d = []
    for k,p in P_ijk.items():
        p2d += [tf.reduce_sum(p,axis=1),tf.reduce_sum(p,axis=2),tf.reduce_sum(p,axis=3)]
    p2d = tf.reduce_mean(p2d, axis=0)

    return cce_sat,cce_consist,p2d



def probe_bsite_tf_frag(pred, bsite, idx, beta, L):
    
    '''
    -------------------------------------------------------------------------
    - pred[?,L,L,130]    : trRosetta predictions (tf.tensor)
    - bsite[m,m,130]     : 1-hot encoded binding site, 6D coords (np.array)
    - idx[m]             : indices of bs residues (np.array, int)
    - beta               : inverse temperature, a scalar (float)
    - L
    -------------------------------------------------------------------------
    '''

    # extend 2D arrays so that each pixel stores the full set coordinates (6D)
    pred = tf.concat([pred,tf.transpose(pred[:,:,:,74:],[0,2,1,3])], axis=-1)
    bsite_np = np.concatenate([bsite,np.transpose(bsite[:,:,74:],[1,0,2])], axis=-1)
    
    # hash function to map pairs and triples of indices to an integer
    hash_ = lambda t : hash(np.sort(t).tostring())

    # number of binding site fragments and their span
    j = np.cumsum([0]+[idx[i]>idx[i-1]+1 for i in range(1,idx.shape[0])])
    n = j[-1]+1
    
    # enumerate all possible pairs and triples
    pairs   = [[i,j] for i in range(n-1) for j in range(i+1,n)]
    #triples = [[i,j,k] for i in range(n-2) for j in range(i+1,n-1) for k in range(j+1,n)]

    # satisfaction score (NEW)
    mask_i = [np.where(j==k)[0] for k in range(n)]
    bs_c6d_ij = {hash_([i,j]): tf.constant(bsite_np[np.ix_(mask_i[i],mask_i[j].T)],dtype=tf.float32)[:,:,:,tf.newaxis]/
                 (mask_i[i].shape[0]*mask_i[j].shape[0]*6)
                 for i in range(n) for j in range(i,n)}

    # pad predicted 6D coords w/ no contact bins
    # for proper handling of CCE calculations near the edges
    max_pad = np.max([mi.shape[0]//2 for mi in mask_i])
    pad = np.array([0]*36+[1] + [0]*36+[1] + [0]*36+[1] + [0]*18+[1] + [0]*36+[1] + [0]*18+[1]).reshape((1,1,1,186))
    pad1 = tf.constant(np.tile(pad,(1,L,max_pad,1)),dtype=tf.float32)
    pad2 = tf.constant(np.tile(pad,(1,max_pad,L+2*max_pad,1)),dtype=tf.float32)
    
    pred_pad = tf.concat([pad2,tf.concat([pad1,pred,pad1],axis=2),pad2],axis=1)
    
    pred_log = tf.math.log(pred_pad+1e-8)
    conv_ = lambda p : tf.nn.conv2d(pred_log,
                                    bs_c6d_ij[hash_(p)],
                                    strides=[1,1,1,1],
                                    padding="SAME")[:,max_pad:-max_pad,max_pad:-max_pad,0]

    # satisfaction score
    CCE_i = [-tf.diag_part(conv_([i,i])[0])[tf.newaxis,:] for i in range(n)]
    
    penalty = {k: signal.convolve2d(np.eye(L), np.full(bs.shape[0:2],1), boundary='wrap', mode='same') #/np.max(bs.shape[0:2])
               for k,bs in bs_c6d_ij.items()}
    
    CCE_ij  = {hash_(p): (-conv_(p) + CCE_i[p[0]][:,:,None] + CCE_i[p[1]][:,None,:])/3 +
                         tf.constant(penalty[hash_(p)],dtype=tf.float32)[tf.newaxis,:,:] for p in pairs}

    # special case - 2 frags
    if len(pairs)==1:
        beta = tf.cast(beta,dtype=tf.float64)
        cce = tf.cast(CCE_ij[hash_(pairs[0])],dtype=tf.float64)
        cce_exp = tf.math.exp(-beta*cce)
        Z = tf.reduce_sum(cce_exp,axis=(1,2))
        p = cce_exp/Z[:,None,None]
        cce_sat = tf.reduce_sum(p*cce, axis=(1,2))
        cce_consist = 0.0
        return tf.cast(cce_sat,dtype=tf.float32), tf.cast(cce_consist,dtype=tf.float32), p


    #CCE_ij  = {hash_(p): -conv_(p) for p in pairs}
    triples = [[i,j,k] for i in range(n-2) for j in range(i+1,n-1) for k in range(j+1,n)]
    CCE_ijk = {hash_(t): CCE_ij[hash_([t[0],t[1]])][:,:,:,None]/3 + \
                         CCE_ij[hash_([t[0],t[2]])][:,:,None,:]/3 + \
                         CCE_ij[hash_([t[1],t[2]])][:,None,:,:]/3 for t in triples}
    
    #CCE_ij_v = [-conv_(p) for p in pairs]


    P_ijk   = {k: tf.math.exp(-tf.cast(beta*cce,dtype=tf.float64))/tf.reduce_sum(tf.math.exp(-tf.cast(beta*cce,dtype=tf.float64)),axis=(1,2,3))[:,None,None,None] for k,cce in CCE_ijk.items()}
    cce_sat = tf.reduce_mean([tf.reduce_sum(cce*tf.cast(P_ijk[k],dtype=tf.float32),axis=(1,2,3)) for k,cce in CCE_ijk.items()],axis=0) 

    # consistency score
    if n > 3:
        cce_consist = []
        for p in np.array(pairs):
            diff = np.setdiff1d(np.arange(n),p)
            for i in range(n-3):
                pi = tf.reduce_sum(P_ijk[hash_([p[0],p[1],diff[i]])], axis=np.sum(p<diff[i])+1)
                for j in range(i+1,n-2):
                    pj = tf.reduce_sum(P_ijk[hash_([p[0],p[1],diff[j]])], axis=np.sum(p<diff[j])+1)
                    cce = 0.5*(-tf.reduce_sum(pi*tf.math.log(pj+1e-8),axis=(1,2))-tf.reduce_sum(pj*tf.math.log(pj+1e-8),axis=(1,2)))
                    cce_consist.append(cce)
        cce_consist = tf.reduce_mean(cce_consist, axis=0)
    else:
        cce_consist = 0.0
    
    # 3-site probabilities projected down to 2d and
    # averaged over all projections and triples
    p2d = []
    for k,p in P_ijk.items():
        p2d += [tf.reduce_sum(p,axis=1),tf.reduce_sum(p,axis=2),tf.reduce_sum(p,axis=3)]
    p2d = tf.reduce_mean(p2d, axis=0)

    return tf.cast(cce_sat,dtype=tf.float32), tf.cast(cce_consist,dtype=tf.float32), p2d
