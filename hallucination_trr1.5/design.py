#!/software/conda/envs/tensorflow/bin/python
import warnings, logging, os, sys
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, os
import json
import numpy as np
from scipy import stats
import networkx as nx
import argparse, pickle
from itertools import permutations
import tensorflow as tf

from resnet import network,load_weights
from bsite import probe_bsite_tf_frag
from utils import *

BKG_DIR = '/home/jue/trDesign/imprint_bs_v06/bkg/'

def main(argv):
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True)
    )

    ########################################################
    # 0. process inputs
    ########################################################
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument("--pdb", '-p', type=str, required=True, help="input PDB with binding site residues")
    p.add_argument("--out", '-o', type=str, required=True, help="output file path and prefix")
    p.add_argument('--len', '-l', type=str, default='100',  help='sequence length')
    p.add_argument('--num', '-n', type=int, default=1,  help='number of models to generate')
    p.add_argument('--start_num', type=int, default=0,  help='index of first model filename')
    p.add_argument('--msa_num', type=int, default=1000, help='number of sequences in MSA being designed. Setting to 1 means single-sequence design')

    p.add_argument('--contigs', type=str, required=False, default=None, help='Regions to constrain, as comma-delimited numeric ranges (e.g. "3-10,24-32")')
    p.add_argument('--mask', type=str, required=False, default=None, help='Regions to constrain (preceded by chain letter), alternating with length ranges of intervening regions (without chain letter). e.g. 0-5,A31-45,25-30,A60-81.')
    p.add_argument('--cs_method', type=str, default='ia', help='method to place contigs. {ia, random}')
    p.add_argument('--min_gap', type=int, help='when --cs_method=random, the shortest loop length (in AAs) to put between contigs.')
    p.add_argument('--keep_order', default=False, type=str2bool, help='when --cs_method=random, whether to keep contigs in input order. Default: False')

    p.add_argument('--loss_pdb', type=float, default=1.0, help='weight for the pdb loss')
    p.add_argument('--loss_bkg', type=float, default=1.0, help='weight for the KL loss')
    p.add_argument('--loss_sat', type=float, default=1.0, help='weight for the satisfaction term')
    p.add_argument('--loss_consist', type=float,default=1.0, help='weight for the consistency term')
    p.add_argument('--loss_clash', type=float, default=1.0, help='weight for the clash term')

    p.add_argument('--cce_cutoff',    default=19.9, type=float, help='filter cce to CB ≤ x')
    p.add_argument('--feat_drop', type=float, default=0, help='dropout rate for the input features')
    p.add_argument('--opt_rate', type=float, default=0.1, help='NGD minimization step size')
    p.add_argument('--opt_iter', type=int, default=200, help='number of minimization steps')
    p.add_argument('--beta0', type=float, dest='b0', default=2.0, help='inverse temperature at the beginning of the minimization')
    p.add_argument('--beta1', type=float, dest='b1', default=20.0, help='inverse temperature at the end of the minimization')
    p.add_argument('--sample', default=True, type=str2bool, help='perform NGD+sample')
    p.add_argument('--seq_hard', default=True, type=str2bool, help='discretize logits before forward pass')
    p.add_argument('--init_sd', type=float, default=0.1, help='std dev of noise to add at logit initiliaztion')
    p.add_argument('--n_models', type=int, default=3, help='number of trRosetta models to use')
    p.add_argument('--double_asym_feat', default=False, type=str2bool, help='double-count theta and phi and divide all output features by 6 (Default: False)')
    o = p.parse_args()

    # check for valid arguments
    if o.cs_method != 'ia' and o.cs_method != 'random':
        sys.exit('--cs_method must be either "ia" or "random"')
    if o.contigs is None and o.mask is None:
        sys.exit('Either --contigs or --mask must be provided.')
    if o.mask is not None and o.len is not None:
        print('Ignoring --len argument for mask mode.')

    # convert paths to absolute paths
    o.pdb = os.path.abspath(o.pdb)
    o.out = os.path.abspath(o.out)

    # write settings file
    print(vars(o),file=open(o.out+'.set','w'))
    print(vars(o))
 
    # read checkpoint file if it exists
    f_checkpoint = f'{o.out}.chkp'
    if os.path.exists(f_checkpoint):
        with open(f_checkpoint, 'r') as f_in:
            hals_done = f_in.readlines()
            last_completed_hal = int(hals_done[-1]) if len(hals_done) != 0 else o.start_num - 1
            if last_completed_hal + 1 == o.start_num + o.num:
                print('All jobs have previously completed. There is nothing more to hallucinate.')
    else:
        last_completed_hal = o.start_num - 1

    # parse pdb
    print(f'extracting features from pdb: {o.pdb}')
    chains = list(set(re.findall(r'[A-Z]', o.contigs if o.contigs is not None else o.mask)))
    pdb_out = prep_input(o.pdb, chain=chains, double_asym_feat=o.double_asym_feat)
    pdb_feat = pdb_out['feat'][None]
    desired_feat = np.copy(pdb_feat)
    pdb_idx = pdb_out['pdb_idx']

    # CCE cutoff
    if o.cce_cutoff is not None:
      pdb_dist = pdb_out["dist_ref"][None]
      mask_cce = np.logical_or(pdb_dist > o.cce_cutoff, pdb_dist < 1.0)
      desired_feat[mask_cce] = 0

    ########################################################
    # setup network
    ########################################################
    
    # load network parameters
    with open('/home/aivan/for/GyuRie/trRosetta2/training/params.json') as jsonfile:
        params = json.load(jsonfile)

    # load network weights
    weights = load_weights('/home/aivan/for/GyuRie/trRosetta2/training/models',params, o.n_models)

    # setup computation graph
    x = tf.placeholder(dtype=tf.float32, shape=(None,None,20))
    drop_rate = tf.placeholder(dtype=tf.float32, shape=())
    p2d,p3d = network(tf.pad(x,[[0,0],[0,0],[0,1]]),*weights,params,drop_rate,o.double_asym_feat)
    p2d = p2d[0]
    p3d = p3d[0]

    ########################################################
    # generate background distributions
    ########################################################
    
    # background distributions are generated by passing 
    # multiple random MSAs through the trRosetta network
    # and averaging over the runs:
    NSEQ_BKG = 5    # number of random sequences in the MSA
    NRUN_BKG = 100  # number of runs

    if o.contigs is not None:
        if '-' in o.len:
            min_L, max_L = [int(n) for n in o.len.split('-')]
        else:
            min_L, max_L = int(o.len), int(o.len)
    elif o.mask is not None:
        min_L, max_L = 0,0
        for el in o.mask.split(','):
            if el[0].isalpha():
                # adding a fixed length contig
                s,e = [int(x) for x in el[1:].split('-')]
                min_L += e - s + 1
                max_L += e - s + 1
            else:
                # adding a variable length gap
                n1,n2 = [int(x) for x in el.split('-')]
                min_L += n1
                max_L += n2
    L_range = np.arange(min_L, max_L+1)

    # generate background distributions
    bkg = {}
    for L in L_range:
        bkg_np = np.zeros((L,L,130))
        sess = tf.Session(config=config)
        fn = os.path.join(BKG_DIR, f'bkg_{L}.npz')
        if os.path.exists(fn):
            print(f"loading precomputed background len={L}")
            bkg_np = np.load(fn)['bkg']
            # these were precomputed with 4 output DDoFs
            if o.double_asym_feat:
                bkg_np = np.concatenate([bkg_np, np.transpose(bkg_np[:,:,74:],[1,0,2])], -1)
        else:
            print(f"generating background len={L}")
            for i in range(NRUN_BKG):
                pssm = np.random.normal(0,o.init_sd,size=(NSEQ_BKG,o.len,20))
                x_inp = argmax(pssm)
                p2d_ = sess.run(p2d, feed_dict={x:x_inp,drop_rate:0.0})
                bkg_np += np.mean(p2d_,axis=0)
            bkg_np /= NRUN_BKG
        bkg[L] = bkg_np

    #####################################################
    # set up losses
    #####################################################

    # background probabilities 
    bkg_tf = tf.placeholder(dtype=tf.float32, shape=(None,None,None))

    n_out_feat = 6 if o.double_asym_feat else 4

    if o.contigs is not None and o.cs_method == 'ia':
        # prepare binding site
        contigs = parse_contigs(o.contigs, pdb_idx)
        bsite_idx = np.array(cons2idxs(contigs))
        bsite = pdb_out['feat'][bsite_idx[:,None], bsite_idx[None,:]]
        bsite_nres  = bsite_idx.shape[0]
        bsite_nfrag = len(contigs)
        
        # fragment sizes and residue indices
        j = np.cumsum([0]+[bsite_idx[i]>bsite_idx[i-1]+1 for i in range(1,bsite_nres)])
        fs = np.array([np.sum(j==i) for i in range(j[-1]+1)])
        fi = [np.where(j==i)[0] for i in range(j[-1]+1)]

        # annealing parameters for ivan contig loss
        beta_start = o.b0
        beta_shift = (o.b1-o.b0)/o.opt_iter
        
        # placeholder for the inverse temperature (for bs imprinting)
        beta = tf.placeholder(dtype=tf.float32, shape=())

        # placeholder for length
        #L_tf = tf.placeholder(dtype=tf.int16, shape=())

        # hallucination loss: KL-divergence of predicted probability 
        # distributions for dist,omega,theta,phi from background
        kl = -tf.math.reduce_sum(p2d*tf.math.log(p2d/bkg_tf),-1)
        kl = tf.math.reduce_mean(kl) / n_out_feat

        # probe b-site
        loss_sat,loss_consist,bsite_ij = probe_bsite_tf_frag(tf.expand_dims(p2d,axis=0),bsite,bsite_idx,beta,min_L,o.double_asym_feat)
        losses = [kl,loss_sat[0],loss_consist]
        loss_labels = ['bkg','sat','consist']
        loss = o.loss_bkg*kl + o.loss_sat*loss_sat[0] + o.loss_consist*loss_consist

    elif o.mask is not None or (o.contigs is not None and o.cs_method == 'random'):
        pdb_feat = tf.placeholder(dtype=tf.float32, shape=(None,None,None))
        mask_pdb = tf.reduce_sum(pdb_feat,-1) / n_out_feat  # 1 where cce loss should be applied
        
        # cross entropy loss for fixed backbone locations
        pdb_loss = -tf.reduce_sum(pdb_feat*tf.log(p2d+1e-8),-1) * mask_pdb / n_out_feat
        pdb_loss = tf.reduce_sum(pdb_loss,[-1,-2]) / (tf.reduce_sum(mask_pdb,[-1,-2])+1e-8)
       
        # kl loss for hallucination 
        # invert the pdb mask
        mask_bkg = 1 - mask_pdb
        mask_bkg *= (1 - tf.eye(tf.shape(mask_bkg)[1]))  # Exclude the diagonal
        
        bkg_loss = -tf.reduce_sum(p2d * tf.log(p2d/(bkg_tf+1e-8)+1e-8),-1) * mask_bkg / n_out_feat
        bkg_loss = tf.reduce_sum(bkg_loss,[-1,-2])/(tf.reduce_sum(mask_bkg,[-1,-2])+1e-8)

        losses = [pdb_loss, bkg_loss]
        loss_labels = ['pdb','bkg']
        loss = o.loss_pdb*pdb_loss + o.loss_bkg*bkg_loss

    # clash score
    if o.loss_clash > 0:
        p3dT = tf.transpose(p3d,[0,1,3,2])
        p3d_top2,_ = tf.nn.top_k(p3dT, k=2)
        clash = tf.reduce_sum((p3d_top2[:,:,:-1,0]*p3d_top2[:,:,:-1,1])**2)
        losses.append(clash)
        loss_labels.append('clash')
        loss = loss + o.loss_clash*tf.nn.relu(clash-2.0)

    # calculate gradient of the total loss with respect to the input MSA (x)
    print('setting up computation graph')
    grad = tf.keras.backend.gradients(loss, x)[0]

    ########################################################
    # hallucination
    ########################################################

    # generate o.num designs
    for n in range(last_completed_hal + 1, o.start_num+o.num):

        if o.mask is not None:
            feat_hal, mappings = apply_mask(o.mask, pdb_out)
            L = feat_hal.shape[1]
        elif o.contigs is not None and o.cs_method == 'random':
            feat_hal, mappings = scatter_contigs(o.contigs, pdb_out, L_range=o.len, 
                                                 keep_order=o.keep_order, min_gap=o.min_gap)
        else:
            L = np.random.randint(min_L,max_L+1)
        if o.contigs is not None and o.cs_method == 'ia':
            b=beta_start

        print(f'Generating output {n} (length {L}):')

        # length-adjusted learning rate
        lr = o.opt_rate * np.sqrt(L)

        # optimize
        pssm = np.random.normal(0,o.init_sd,size=(o.msa_num,L,20))
        for k in range(1,o.opt_iter+1):

            x_inp = sample(pssm, o.seq_hard) if o.sample else argmax(pssm) 

            if o.contigs is not None and o.cs_method == 'ia':
                feed_dict = {x:x_inp, bkg_tf:bkg[L], drop_rate:o.feat_drop, beta:b}
            elif o.mask is not None or (o.contigs is not None and o.cs_method == 'random'):
                feed_dict = {x:x_inp, bkg_tf:bkg[L], drop_rate:o.feat_drop, pdb_feat:feat_hal[0]}
                
            grad_,ls_,l_ = sess.run([grad,losses,loss], feed_dict=feed_dict)

            if o.contigs is not None and o.cs_method == 'ia': b+=beta_shift
            grad_ = grad_/np.linalg.norm(grad_, axis=(1,2), keepdims=True) + 1e-8
            pssm -= lr * grad_

            if k%10==0:
                losses_string = ' '.join([f'{x:6.3f}' for x in ls_])
                print(f'\tstep {k:4}: [ {", ".join(loss_labels)} ]=[ {losses_string} ] total_loss={l_:6.3f}')

        # final contig placements from ivan method, save results
        if o.contigs is not None and o.cs_method == 'ia':
            # recalculate w/o dropout
            feed_dict.update({x:argmax(pssm), drop_rate:0})
            loss_,losses_,p2d_,p3d_,bs_ij_ = sess.run([loss,losses,p2d,p3d,bsite_ij], 
                                                      feed_dict=feed_dict)

            # check binding site
            i2 = np.argsort(bs_ij_[0].flatten())[-(bsite_nfrag**2-bsite_nfrag):]
            G = nx.Graph()
            G.add_nodes_from([i for i in range(L)])
            G.add_edges_from([(i%L,i//L) for i in i2])
            max_clique_size = nx.algorithms.max_weight_clique(G,weight=None)[1]

            if max_clique_size > bsite_nfrag:
                max_clique_size = bsite_nfrag

            max_cliques = [c for c in nx.algorithms.enumerate_all_cliques(G) 
                           if len(c)==max_clique_size]

            mtx = np.sum(np.sum((p3d_[:,:,None,:,:-1]*
                                 p3d_[:,:,:,None,:-1])**2,axis=-1),
                         axis=(0,1))
            np.fill_diagonal(mtx,0)

            print("bs size: %d out of %d"%(max_clique_size, bsite_nfrag), max_cliques)

            # enumerate all possible fragment orders and
            # identify the best scoring one
            trials = []
            for clique in max_cliques:
                for p in permutations(range(len(fs)),max_clique_size):
                    p = np.array(p)
                    a = np.hstack([np.arange(j)+i-(j-1)//2 for i,j in zip(clique,fs[p])])

                    # skip if out of sequence range
                    if np.sum(a<0)>0 or np.sum(a>=L)>0:
                        continue

                    # skip if fragments clash
                    if np.unique(a).shape[0]!=a.shape[0]:
                        continue

                    b = np.hstack([fi[i] for i in p])

                    P = p2d_[np.ix_(a,a.T)]
                    Q = bsite[np.ix_(b,b.T)]
                    s = np.mean(np.sum(-np.log(P)*Q,axis=-1))/4
                    trials.append((s,a,b,p))

            trials.sort(key=lambda x: x[0])

            if (len(trials) ==0) or (max_clique_size != bsite_nfrag):
                print('No motif placements could be found :(')
            else:
                best=trials[0]
                zscore = stats.zscore([t[0] for t in trials])[0]
                print("best: score= %.5f   zscore= %.5f   trials= %d   order="%(best[0],zscore,len(trials)), best[3])
                sys.stdout.flush()

                seq = idx2aa(pssm[...,:20].argmax(-1)[0])
                print(seq)

                name = f'{o.out}_{n}'

                scores = np.hstack([ls_,[l_,np.sum(mtx),best[0],zscore]])
                np.savez_compressed(name,
                                    dist=p2d_[:,:,:37],
                                    omega=p2d_[:,:,37:37*2],
                                    theta=p2d_[:,:,37*2:37*3],
                                    phi=p2d_[:,:,37*3:],
                                    idx_pred=best[1],
                                    idx_bsite=best[2],
                                    msa=pssm[...,:20].argmax(-1),
                                    p2d=bs_ij_,
                                    scores=scores)

                with open(f'{name}.fas','w') as f:
                    f.write(">%s\n%s\n"%(name,seq))

                outdict = {}
                outdict['con_ref_pdb_idx'] = [pdb_idx[bsite_idx[i]] for i in best[2]]
                outdict['con_hal_pdb_idx'] = [('A',i+1) for i in best[1]]
                outdict['loss_nodrop'] = {'bkg':losses_[0], 'sat':losses_[1], 
                                          'consist':losses_[2],'clash':losses_[3], 'pdb':best[0]}
                outdict['settings'] = vars(o)
                with open(f'{name}.trb', 'wb') as outf:
                    pickle.dump(outdict, outf)

        elif o.mask is not None or (o.contigs is not None and o.cs_method == 'random'):
            # recalculate w/o dropout
            feed_dict.update({x:argmax(pssm), drop_rate:0})
            loss_,losses_,p2d_ = sess.run([loss,losses,p2d], feed_dict=feed_dict)

            name = f'{o.out}_{n}'
            np.savez_compressed(f'{name}.npz',
                                dist=p2d_[:,:,:37],
                                omega=p2d_[:,:,37:37*2],
                                theta=p2d_[:,:,37*2:37*3],
                                phi=p2d_[:,:,37*3:])

            seq = idx2aa(pssm[...,:20].argmax(-1)[0])
            with open(f'{name}.fas','w') as f:
                print(f'>{name}\n{seq}', file=f)

            outdict = {'loss_nodrop':dict(zip(loss_labels,losses_)), 'settings':vars(o)}
            outdict.update(mappings)

            with open(f'{name}.trb', 'wb') as outf:
                pickle.dump(outdict, outf)

        # record completed design numbers in the checkpoint file
        with open(f_checkpoint, 'a+') as f_out:
            print(n,file=f_out)
        
if __name__ == '__main__':
    main(sys.argv[1:])

