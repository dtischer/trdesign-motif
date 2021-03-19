import numpy as np
from scipy.special import softmax

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

def sample(logit, seq_soft=False):
    NSEQ,NRES,NS = logit.shape
    eps=1e-20
    U = np.random.uniform(size=logit.shape)
    pssm_sampled = softmax(logit-np.log(-np.log(U+eps)+eps),axis=-1)
    if seq_soft:
        x_inp = pssm_sampled
    else:
        x_inp = np.eye(20)[pssm_sampled[...,:20].argmax(-1)]
    return x_inp.reshape((NSEQ,NRES,20))
