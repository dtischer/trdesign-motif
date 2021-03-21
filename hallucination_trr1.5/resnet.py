import os
import numpy as np
import tensorflow as tf
from mrf import MRF1,MRF

# a function for loading network weights
def load_weights(DIR, params, n_models=3):

    w,b = [],[]
    beta,gamma = [],[]

    nconv2d  = (params['CYCLS1']+params['CYCLS2'])*len(params['DRATES'])*2+9
    nlaynorm = nconv2d-6

    for filename in sorted(os.listdir(DIR)):

        if not filename.endswith(".index"):
            continue
        if not filename.startswith("model.step2"):
            continue

        mname = DIR+"/"+os.path.splitext(filename)[0]
        print('reading weights from:', mname)

        w.append([
            tf.train.load_variable(mname, 'conv2d/kernel')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
            for i in range(nconv2d)])

        b.append([
            tf.train.load_variable(mname, 'conv2d/bias')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
            for i in range(nconv2d)])

        beta.append([
            tf.train.load_variable(mname, 'InstanceNorm/beta')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
            for i in range(nlaynorm)])

        gamma.append([
            tf.train.load_variable(mname, 'InstanceNorm/gamma')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
            for i in range(nlaynorm)])

        if len(w)==n_models: break

    return (w,b,beta,gamma)

#
# auxiliary functions for ResNet
#
def InstanceNorm(features,beta,gamma):
    mean,var = tf.nn.moments(features,axes=[1,2])
    x = (features - mean[:,None,None,:]) / tf.sqrt(var[:,None,None,:]+1e-5)
    out = tf.constant(gamma)[None,None,None,:]*x + tf.constant(beta)[None,None,None,:]
    return out

def Conv2d(features,w,b,d=1):
    x = tf.nn.conv2d(features,tf.constant(w),strides=[1,1,1,1],padding="SAME",dilations=[1,d,d,1]) + tf.constant(b)[None,None,None,:]
    return x

Activation = tf.nn.elu

def resblock(layers2d,w,b,beta,gamma,dilation,i,j):
    layers2d.append(Conv2d(layers2d[-1],w[i],b[i],dilation))
    layers2d.append(InstanceNorm(layers2d[-1],beta[j],gamma[j]))
    layers2d.append(Activation(layers2d[-1]))
    layers2d.append(Conv2d(layers2d[-1],w[i+1],b[i+1],dilation))
    layers2d.append(InstanceNorm(layers2d[-1],beta[j+1],gamma[j+1]))
    layers2d.append(Activation(layers2d[-1] + layers2d[-6]))

def add_resblock(layers2d,w,b,beta,gamma,dilation,i,j):
    layers2d.append(resblock(layers2d[-1],w,b,beta,gamma,dilation,i,j))


# ResNet
def network(x,w,b,beta,gamma,params,drop_rate):

    #inputs = MRF1(x,params)
    nrow = tf.shape(x)[0]
    inputs = tf.cond(nrow>1, lambda: MRF(x, params), lambda: MRF1(x,params))
    
    # add dropout to features (don't apply dropout to diagonal)
    e = tf.eye(tf.shape(inputs)[1])[None,:,:,None]
    inputs_drop = e*inputs + (1-e)*tf.nn.dropout(inputs,rate=drop_rate)

    # lists to store separate branches and their predictions
    layers2d = [[] for _ in range(len(w))]
    preds = [[] for _ in range(5)] # theta,phi,dist,omega,grids3d

    # create a separate branch for every checkpoint
    for i in range(len(w)):

        # project features down
        layers2d[i] = [inputs_drop]
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][0],b[i][0]))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][0],gamma[i][0]))
        layers2d[i].append(Activation(layers2d[i][-1]))

        # first cycle with more filters
        k = 1
        for _ in range(params['CYCLS1']):
            for dilation in params['DRATES']:
                resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],dilation,k,k)
                k += 2

        # project down
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],1))
        layers2d[i].append(Activation(layers2d[i][-1]))
        k += 1

        # second cycle with less filters
        for _ in range(params['CYCLS2']):
            for dilation in params['DRATES']:
                resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],dilation,k,k-1)
                k += 2

        # one more block with dilation=1
        resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],1,k,k-1)

        # probabilities for theta, phi and 3d grids
        preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-5],b[i][-5])))
        preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-4],b[i][-4])))
        preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-3],b[i][-3])))

        # symmetrize
        layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

        # probabilities for dist and omega
        preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-2],b[i][-2])))
        preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-1],b[i][-1])))

    # average over all branches
    pt  = tf.reduce_mean(tf.stack(preds[0]),axis=0)
    pp  = tf.reduce_mean(tf.stack(preds[1]),axis=0)
    pd  = tf.reduce_mean(tf.stack(preds[3]),axis=0)
    po  = tf.reduce_mean(tf.stack(preds[4]),axis=0)

    # concatenated 2D predictions for dist,omega,theta,phi
    p2d = tf.concat([pd,po,pt,pp],axis=-1)

    # stacked predictions for 3D grids
    p3d = tf.stack(preds[2])
    
    # transpose p3d to make batch dimension to be first
    p3d = tf.transpose(p3d,[1,0,2,3,4])

    return p2d,p3d
