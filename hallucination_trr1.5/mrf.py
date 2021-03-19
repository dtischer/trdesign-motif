import numpy as np
import tensorflow as tf

# extract input features from an msa ('x' is an input MSA)
def MRF(x, params):

    N,L,A = [tf.shape(x)[k] for k in range(3)]
    F = L*A

    lam=4.5
    lid=0.3
    lid_scale=18.0
    use_entropy=False
    
    with tf.name_scope('reweight'):
        if lid > 0.0:
            # experimental option to downweight distant sequences
            id_len = tf.cast(L, tf.float32)
            id_mtx = tf.tensordot(x,x,[[1,2],[1,2]]) / id_len
            id_mask = tf.sigmoid((lid-id_mtx)*lid_scale)
            weights = 1.0/(tf.reduce_sum(id_mask,-1)+1.0)
        else:
            # give each sequence equal weight
            weights = tf.ones(N)

    with tf.name_scope('covariance'):
        # compute covariance matrix of the msa
        x_flat = tf.reshape(x, (N,F))
        num_points = tf.reduce_sum(weights)
        one = tf.reduce_sum(tf.square(weights))/num_points
        x_mean = tf.reduce_sum(x_flat * weights[:,None], axis=0, keepdims=True) / num_points
        x_flat = (x_flat - x_mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x_flat),x_flat)/(num_points - one)

    with tf.name_scope('inv_convariance'):
        # compute the inverse of the covariance matrix
        I_F = tf.eye(F)
        rm_diag = 1-tf.eye(L)
        cov_reg = cov + I_F * lam/tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg + tf.random.uniform(tf.shape(cov_reg)) * 1e-8)

        x1 = tf.reshape(inv_cov,(L,A,L,A))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (L,L,A*A))

        # extract contacts
        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * rm_diag
        x3_ap = tf.reduce_sum(x3,0)
        x4 = (x3_ap[None,:] * x3_ap[:,None]) / tf.reduce_sum(x3_ap)
        contacts = (x3 - x4) * rm_diag

        # 2-site entropies and gaps
        f_ij = tf.tensordot(weights[:,None,None]*x, x, [[0],[0]]) / num_points + 1e-9
        #gaps = f_ij[:,20,:,20]
        gaps = tf.zeros((L,L,1))

        if use_entropy:
            h_ij = tf.reduce_sum( -f_ij * tf.log(f_ij), axis=[1,3], keepdims=True)
        else:
            h_ij = tf.zeros((L,L,1))

        # sequence separation
        idx = tf.range(L)
        seqsep = tf.abs(idx[:,None]-idx[None,:])+1
        seqsep = tf.log(tf.cast(seqsep,dtype=tf.float32))
    
        # combine 2D features
        feat_2D = tf.concat([features, contacts[:,:,None], h_ij, gaps, seqsep[:,:,None]], axis=-1)

    with tf.name_scope('1d_features'):
        # sequence
        x_i = tf.stop_gradient(x[0,:,:20])
        # pssm
        f_i = tf.reduce_sum(weights[:,None,None] * x, axis=0) / num_points
        # entropy
        if use_entropy:
            h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
        else:
            h_i = tf.zeros((L,1))
        # non-gaps (assume that there are none)
        #n_i = tf.log(tf.reduce_sum(msw[:,:,:20],axis=[0,2]))
        n_i = tf.fill((L,1),tf.log(tf.cast(N,tf.float32)))

        # tile and combine 1D features
        feat_1D = tf.concat([x_i,f_i,h_i,n_i], axis=-1)
        feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
        feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    # combine 1D and 2D features
    feat = tf.concat([feat_1D_tile_A,
                      feat_1D_tile_B,
                      feat_2D], axis=-1)

    return tf.reshape(feat, [1,L,L,2*43+445])


# extract input features from a single sequence
def MRF1(x, params):

    # dimensions
    N,L,A = [tf.shape(x)[k] for k in range(3)]

    # sequence separation
    idx = tf.range(L)
    seqsep = tf.abs(idx[:,None]-idx[None,:])+1
    seqsep = tf.log(tf.cast(seqsep,dtype=tf.float32))

    # collect features
    f_i = x 
    h_i = tf.zeros((N,L))
    n_i = tf.zeros((N,L)) # does this feature matter???
    #n_i = tf.fill((N,L),tf.log(tf.cast(N,tf.float32)+1e-8)) 
                        
    f1d = tf.concat([x[:,:,:20], f_i, h_i[:,:,None], n_i[:,:,None]], axis=-1)
                            
    inputs = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,L,1]),
                        tf.tile(f1d[:,None,:,:], [1,L,1,1]),
                        tf.tile(tf.eye(L)[None,:,:,None], [N,1,1,441])/params['DCAREG'],
                        tf.zeros([N,L,L,3], tf.float32),
                        tf.tile(seqsep[None,:,:,None],[N,1,1,1])], axis=-1)

    return tf.reshape(inputs, [N,L,L,2*43+445])
