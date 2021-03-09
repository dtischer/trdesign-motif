import numpy as np
import copy
from scipy import signal

# Should work in both TF1 and TF2
#from keras.utils.vis_utils import plot_model
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v1.keras.backend as K1
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Lambda, Layer, Concatenate, Average, Dropout
import tensorflow.keras.backend as K
tf.compat.v1.disable_v2_behavior()
tf1.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

# HACK to fix compatibility issues with RTX2080
config = tf1.ConfigProto()
config.gpu_options.allow_growth = True

from utils import *
###################################################
# RESNET
###################################################
# custom layers
class instance_norm(Layer):
  def __init__(self, axes=(1,2),trainable=True):
    super(instance_norm, self).__init__()
    self.axes = axes
    self.trainable = trainable
  def build(self, input_shape):
    self.beta  = self.add_weight(name='beta',shape=(input_shape[-1],),
                                 initializer='zeros',trainable=self.trainable)
    self.gamma = self.add_weight(name='gamma',shape=(input_shape[-1],),
                                 initializer='ones',trainable=self.trainable)
  def call(self, inputs):
    mean, variance = tf.nn.moments(inputs, self.axes, keepdims=True)
    return tf.nn.batch_normalization(inputs, mean, variance, self.beta, self.gamma, 1e-6)

def RESNET(mode="TrR", blocks=12, weights=None, trainable=False, resnet_drop_out=0):
  ## INPUT ##
  if mode == "TrR":
    inputs = Input((None,None,526)) # (batch,len,len,feat)
    A = inputs
  if mode == "TrR_BKG":
    inputs = Input(shape=[],dtype=tf.int32)
    A = Lambda(lambda x: tf.random.normal([1,x[0],x[0],64]))(inputs)

  ex = {"trainable":trainable}
  A = Dense(64, **ex)(A)
  A = instance_norm(**ex)(A)
  A = Activation("elu")(A)

  ## RESNET ##
  def resnet(X, dilation=1, filters=64, win=3):
    Y = Conv2D(filters, win, dilation_rate=dilation, padding='SAME', **ex)(X)
    Y = instance_norm(**ex)(Y)
    Y = Activation("elu")(Y)
    Y = Dropout(rate=resnet_drop_out)(Y, training=(resnet_drop_out>0))  # add dropout here
    Y = Conv2D(filters, win, dilation_rate=dilation, padding='SAME', **ex)(Y)
    Y = instance_norm(**ex)(Y)
    return Activation("elu")(X+Y)

  for _ in range(blocks):
    for dilation in [1,2,4,8,16]: A = resnet(A, dilation)
  A = resnet(A, dilation=1)

  ## OUTPUT ##
  A_asym    = A
  p_theta   = Dense(25, activation="softmax", **ex)(A_asym)
  p_theta_T = tf.transpose(p_theta, (0,2,1,3))
  p_phi     = Dense(13, activation="softmax", **ex)(A_asym)
  p_phi_T   = tf.transpose(p_phi, (0,2,1,3))
  A_sym     = Lambda(lambda x: (x + tf.transpose(x,[0,2,1,3]))/2)(A)
  p_dist    = Dense(37, activation="softmax", **ex)(A_sym)
  #p_bb      = Dense(3,  activation="softmax", **ex)(A_sym)
  p_omega   = Dense(25, activation="softmax", **ex)(A_sym)
  outs      = Concatenate()([p_theta, p_phi, p_dist, p_omega, p_theta_T, p_phi_T])

  ## MODEL ##
  model = Model(inputs, outs)
  if weights is not None: model.set_weights(weights)
  return model

def load_weights(filename):
  weights = [np.squeeze(w) for w in np.load(filename, allow_pickle=True)]
  # remove weights for beta-beta pairing
  del weights[-4:-2]
  return weights

################################################################################################
# Ivan's TrRosetta Background model for backbone design
################################################################################################
def get_bkg(L,DB_DIR=None):
  # get background feat for [L]ength
  K.clear_session()
  K1.set_session(tf1.Session(config=config))
  bkg = {l:[] for l in L}
  bkg_model = RESNET(mode="TrR_BKG", blocks=7)  # why isn't this 12, to match use case?
  for w in range(1,5):
    weights = load_weights(f"{DB_DIR}/bkgr_models/bkgr0{w}.npy")
    bkg_model.set_weights(weights)
    for l in L:
      bkg_model.predict([10])  # what is this doing? a "warm up?"
      bkg[l].append(bkg_model.predict([l])[0])
  return {l:np.mean(bkg[l],axis=0) for l in L}

###############################################################################
# SETUP model to just predict loss and gradients (called by predict.py)
###############################################################################
class mk_predict_model:
  def __init__(self, n_models=5, msa_design=False, diag=(2/9),
               eps=1e-8, DB_DIR=".", serial=False, resnet_drop_out=0):
    self.serial = serial

    K.clear_session()
    K1.set_session(tf1.Session(config=config))
    input_params = {"batch_size":1,"dtype":tf.float32}

    # inputs
    I = Input(shape=(None, None, 21), **input_params)
    I_seq = tf.one_hot(tf.argmax(I[...,:20],-1),20)
    def add_gap(x): return tf.pad(x,[[0,0],[0,0],[0,0],[0,1]])
    if msa_design: I_feat = MRF()(I)
    else: I_feat = PSSM(diag=diag)([I_seq,I])

    # load each model
    self.models = []
    for token in ["xaa","xab","xac","xad","xae"][:n_models]:
      print(f"loading model: {token}")
      weights = load_weights(f"{DB_DIR}/models/model_{token}.npy")
      if self.serial: self.models.append(weights)
      else: self.models.append(RESNET(weights=weights, mode="TrR")(I_feat))

    if self.serial: O_feat = RESNET(mode="TrR", resnet_drop_out=resnet_drop_out)(I_feat)
    else: O_feat = tf.reduce_mean(self.models,0)

    # define model
    self.model = Model(I, O_feat)

  def predict(self, inputs):
    if self.serial:
      preds = []
      for weights in self.models:
        self.model.set_weights(weights)
        preds.append(self.model.predict(inputs))
      return np.mean(preds,0)
    else:
      return self.model.predict(inputs)

##############################################################################
# TrR DESIGN
##############################################################################
class mk_design_model:
  ###############################################################################
  # DO SETUP
  # Returns gradients of inputs wrt loss
  ###############################################################################
  def __init__(self, seq_mode='MSA', feat_drop=None, loss_aa_comp=None,
               loss_pdb=None, loss_bkg=None, loss_eng=None, loss_contacts=None,
               loss_aa_ref=None, loss_keep_out=None, cs_method=None, loss_contigs=None,
               loss_hbn=None,
               serial=False, n_models=5, specific_models=None,
               eps=1e-8, DB_DIR=".",
               kw_MRF={}, kw_PSSM={}, kw_ContigSearch={}, kw_probe_bsite={}, kw_hbnets={}):
    self.serial = serial
    self.feat_drop = feat_drop
      
    # capture intermediate features to output for manual inspection later
    ex_out = {}
    def add_output(term, label):
      ex_out[label] = term

    # reset graph
    K.clear_session()
    K1.set_session(tf1.Session(config=config))

    # configure inputs
    self.inputs_runtime_labels,inputs = [],[]
    def add_input(shape, label, dtype=tf.float32):
      inputs.append(Input(shape, batch_size=1, dtype=dtype))
      self.inputs_runtime_labels.append(label)
      return inputs[-1][0] if len(shape) == 0 else inputs[-1]

    ################################
    # input features
    ################################
    def add_gap(x): return tf.pad(x,[[0,0],[0,0],[0,0],[0,1]])
    
    # constant inputs
    I = add_input((None,None,20),"I")
    loss_weights = add_input((None,),"loss_weights")
    sample = add_input([],"sample",tf.bool)
    hard = add_input([],"seq_hard",tf.bool)
    temp = add_input([],"temp",tf.float32)
    train = add_input([],"train",tf.bool)
    
    # How will we treat the input logits?
    I_soft, I_hard = categorical(I, temp=temp, hard=hard, sample=sample)
    if seq_mode == 'MSA':
      print("mode: msa design")
      I_feat = MRF(**kw_MRF)(add_gap(I_hard))
    elif seq_mode == 'PSSM':
      print("mode: pssm design")
      I_feat = PSSM(**kw_PSSM)([I_hard,add_gap(I_soft)])
    elif seq_mode == 'SEQ':
      print("mode: single sequence design")
      I_feat = PSSM(**kw_PSSM)([I_hard,add_gap(I_hard)])

    # add dropout to features
    if feat_drop > 0 and feat_drop is not None:
      e = tf.eye(tf.shape(I_feat)[1])[None,:,:,None]
      I_feat_drop = tf.nn.dropout(I_feat,rate=feat_drop)
      # exclude dropout at the diagonal
      I_feat_drop = e*I_feat + (1-e)*I_feat_drop
      I_feat = K.switch(train, I_feat_drop, I_feat)

    ################################
    # output features
    ################################
    self.models = []
    tokens = np.array(["xaa","xab","xac","xad","xae"])
    if specific_models is None:
      specific_models = np.arange(n_models)
      
    for token in tokens[specific_models]:
      # load weights (for serial mode) or all models (for parallel mode)
      print(f"loading model: {token}")
      weights = load_weights(f"{DB_DIR}/models/model_{token}.npy")
      if serial: self.models.append(weights)
      else: self.models.append(RESNET(weights=weights, mode="TrR")(I_feat))
        
    if serial: O_feat = RESNET(mode="TrR")(I_feat)
    else: O_feat = tf.reduce_mean(self.models,0)

    ################################
    # define loss
    ################################
    self.loss_label,loss = [],[]
    def add_loss(term,label):
      loss.append(term)
      self.loss_label.append(label)
      
    ################################
    # CE and KL loss for FLEXIBLE contig placement
    ################################
    L = tf.shape(I_feat)[1]
    if loss_contigs:
      if cs_method == 'dt':
        ################################
        # Doug's contig search algorithm
        ################################
        # make inputs on the graph
        # (pdb geometry passed in construction of ContigSearch layer)
        bkg = add_input([None,None,138], 'bkg')
        beta_cs = add_input([], 'beta_cs')

        # search
        kw_cs = {'weight_cce':loss_weights[0,0], 'weight_kl':loss_weights[0,1]}  # is there a less hacky way to do this? Can't reference loss_label because the loss terms have't been added yet!
        cs = ContigSearch(**kw_ContigSearch)
        best_cce_unweighted, best_kl_unweighted, best_con_idxs0, branch_weights, branch_losses, branch_con_idxs0 = cs([O_feat, bkg, beta_cs], **kw_cs)
        add_loss(best_cce_unweighted[None], 'pdb')
        add_loss(best_kl_unweighted[None], 'bkg')
        best_con_idxs0 = tf.stop_gradient(best_con_idxs0)
        branch_con_idxs0 = tf.stop_gradient(branch_con_idxs0)

        # make LUTs for easy conversion between ref and hal idxs
        con_ref = tf.constant(cons2idxs(kw_ContigSearch['contigs']))
        def ref2hal(ref_idx):
          '''
          This is a one to many operation. Want to return hal_idx for all branches
          Think of lut as (ref_idx, branch)

          Returns (branch, hal_idx) tensor
          Each row is what the ref_idx maps to in each branch returned by ContigSearch
          '''
          n_branch = tf.cast(tf.shape(branch_con_idxs0)[0], tf.int32)
          lut_shape = (tf.math.reduce_max(con_ref)+1, n_branch)
          lut = tf.scatter_nd(con_ref[:,None], tf.transpose(branch_con_idxs0, (1,0)), lut_shape)
          hal_idx = tf.gather_nd(lut, ref_idx[:,None])
          return tf.transpose(hal_idx, (1,0))

        # add separate loss term for hbnet cce
        if loss_hbn is not None:
          # reference idx and geo.
          hbn_ref_idx = np.array(cons2idxs(kw_hbnets['contigs']))
          hbn_ref_geo = kw_ContigSearch['ptn_geo'][hbn_ref_idx[:,None], hbn_ref_idx[None,:]]
          
          # hal idx and geo. This has to be done in tf
          hbn_hal_idx = ref2hal(hbn_ref_idx)
          branch, n = tf.shape(hbn_hal_idx)[0], tf.shape(hbn_hal_idx)[1] 
          a = tf.broadcast_to(hbn_hal_idx[:,:,None], (branch,n,n))
          b = tf.broadcast_to(hbn_hal_idx[:,None,:], (branch,n,n))
          gnd_idx = tf.stack([a, b], -1)
          hbn_hal_geo = tf.gather_nd(O_feat[0], gnd_idx)
          add_output(gnd_idx[None], 'gnd_idx')
          add_output(hbn_hal_geo[None], 'hbn_hal_geo')
          
          # calculate cce
          # 1. broadcast ref_geo to all gathered branches of hal_geo
          hbn_ref_geo = tf.broadcast_to(hbn_ref_geo[None], tf.shape(hbn_hal_geo))  # (branch, hbn_idx, hbn_idx, 6D_geo)
          add_output(hbn_ref_geo[None], 'hbn_ref_geo')
          
          # 2. Mask for diagonal and d > 20A
          mask_hbn = tf.reduce_sum(hbn_ref_geo, -1) / 6
          add_output(mask_hbn[None], 'mask_hbn')
          
          # 3. cce at every ij pair
          cce_ij = tf.reduce_sum(-hbn_ref_geo * tf.log(hbn_hal_geo + eps), -1) / 6
          add_output(cce_ij[None], 'cce_ij')
          
          # 4. Mean, unweighted cce of every ContigSearch branch. Excludes diagonal and d > 20
          cce_mean_unweighted = tf.reduce_sum(mask_hbn * cce_ij, (1,2)) / tf.reduce_sum(mask_hbn, (1,2))  # (branch,)
          add_output(cce_mean_unweighted[None], 'cce_mean_unweighted')
          
          # 5. Weighted superposition of all branches
          cce_weighted = cce_mean_unweighted * branch_weights
          add_loss(tf.reduce_sum(cce_weighted)[None], 'hbn')
          
          # track outputs
          add_output(branch_con_idxs0[None], 'branch_con_idxs0')
          add_output(hbn_hal_idx[None], 'hbn_hal_idx')
          add_output(cce_weighted[None], 'hbn_cce_weighted')
          
        # add outputs
        add_output(best_con_idxs0[None], "con_hal_idx0")
        add_output(con_ref[None], "con_ref_idx0")
        add_output(branch_weights[None], 'branch_weights')
        add_output(branch_losses[None], 'branch_losses')
      
      elif cs_method == 'ia':
        ################################
        # Ivan's contig search algorithm
        ################################
        bkg = add_input([None,None,138], 'bkg')
        beta_ia = add_input([None,], 'beta_ia')
        cce_sat, cce_consist, bsite_ij = probe_bsite_tf_frag(O_feat, beta=beta_ia[0,0], **kw_probe_bsite)

        add_loss(cce_sat, 'pdb')
        add_loss(cce_consist[None], 'cce_consist')
        add_output(bsite_ij[None], 'bsite_ij')

        # kl loss over all ij pairs (except the diagonal)
        diag_mask = 1 - tf.eye(tf.shape(O_feat)[1])
        diag_mask = diag_mask[None]  # [batch, L, L]
        bkg_loss = -K.sum(O_feat * K.log(O_feat / (bkg + eps) + eps), -1) / 6
        bkg_loss *= diag_mask
        add_loss(K.sum(bkg_loss,[-1,-2])/(K.sum(diag_mask,[-1,-2])+eps),"bkg")
      
    else:
      ################################
      # CE and KL loss for FIXED contig placement
      ################################
      # starting making the pdb_mask_2D
      if (loss_pdb is not None) or (loss_eng is not None):
        pdb = add_input([None,None,138], 'pdb')  # make the graph input
        pdb_mask_2D = tf.reduce_sum(pdb,-1) / 6
        if add_pdb_mask:
          pdb_mask_2D *= pdb_mask[:,:,None]*pdb_mask[:,None,:]  # NEED TO EXPLICITLY EXCLUDE DIAGONAL
      
      # cross-entropy loss for fixed backbone design
      if loss_pdb is not None:
        pdb_loss = -K.sum(pdb*K.log(O_feat+eps),-1) * pdb_mask_2D / 6
        add_loss(K.sum(pdb_loss,[-1,-2]) / (K.sum(pdb_mask_2D,[-1,-2])+eps),"pdb")
        #add_output(pdb_loss, "pdb_loss_2D")
      
      # minimize statical energy function (family wide hallucinations)
      elif loss_eng is not None:
        eng_loss = -K.sum(O_feat*K.log(pdb+eps), -1) * pdb_mask_2D / 6  # pdb=family constraints
        add_loss(K.sum(eng_loss,[-1,-2])/(K.sum(pdb_mask_2D,[-1,-2])+eps),"eng")
      
      # kl loss for hallucination
      if loss_bkg is not None:
        bkg = add_input([None,None,138], 'bkg')
        bkg_mask_2D = tf.reduce_sum(bkg,-1) / 6
        if add_pdb_mask:
          bkg_mask_2D -= pdb_mask_2D                           # NEED TO EXPLICITLY EXCLUDE DIAGONAL
        bkg_loss = -K.sum(O_feat*K.log(O_feat/(bkg+eps)+eps),-1) * bkg_mask_2D / 6
        add_loss(K.sum(bkg_loss,[-1,-2])/(K.sum(bkg_mask_2D,[-1,-2])+eps),"bkg")

    ################################
    # Common loss functions
    ################################
    # standard amino acid composition loss
    if loss_aa_ref is not None:
      aa = tf.constant(AA_REF, dtype=tf.float32)
      aa_loss = K.sum(K.mean(I_seq*aa,[-2,-3]),-1)
      add_loss(aa_loss,"aa")

    # custom amino acid composition loss
    elif loss_aa_comp is not None:
      aa = tf.constant(AA_COMP, dtype=tf.float32)
      I_aa = K.mean(I_seq,[-2,-3])
      aa_loss = K.sum(I_aa * K.log(I_aa/(aa+eps)+eps),-1)
      add_loss(aa_loss,"aa")
      
    # L1 loss for MSA contacts
    if loss_contacts is not None:
      msa_l1_loss = K.mean(K.abs(I_feat[...,-1]), axis=[-1,-2])
      add_loss(msa_l1_loss, 'contacts')
      add_output(msa_l1_loss, 'contacts')
      
    ###############################
    # add tracked tensors that are not easy to directly access (e.g. buried in a custom layer)
    # ex: add_output(tf1.get_default_graph().get_tensor_by_name("contig_search/strided_slice_1:0")[None], "best_con")
    ###############################
    #g = tf1.get_default_graph()  # this step is VERY slow!!!
    #add_output(g.get_tensor_by_name("contig_search/mk_tree/add_6:0")[None], "loss_1D_2D")
    #add_output(g.get_tensor_by_name("contig_search/mk_tree/add_13:0")[None], "final_loss")
    #add_output(g.get_tensor_by_name("contig_search/mk_tree/Reshape_5:0")[None], "cons")
    #add_output(g.get_tensor_by_name("contig_search/mk_tree/Reshape_6:0")[None], "idxs")
    
    if len(ex_out) == 0: ex_out.update({'dumby': tf.constant([0], dtype=tf.int32)})
    self.ex_out_k, ex_out_v = zip(*ex_out.items())
    
    ################################
    # define gradients
    ################################
    print(f"The loss function is composed of the following: {self.loss_label}")
    loss = tf.stack(loss,-1) * loss_weights
    grad = Lambda(lambda x: tf.gradients(x[0],x[1])[0])([loss, I])

    ################################
    # define model
    ################################
    self.out_label = ['grad', 'loss', 'feat', 'seq_prob'] + list(self.ex_out_k)
    self.model = Model(inputs, [grad,loss,O_feat,I_soft] + list(ex_out_v))

  ###############################################################################
  # DO DESIGN
  # Do the gradient descent steps
  ###############################################################################
  def design(self, weights={}, graph_inputs={}, msa_num=1, rm_aa=None,
             opt_method="GD", opt_iter=100, opt_rate=1.0, opt_decay=2.0,
             temp_soft_seq_decay=2.0, temp_soft_seq_min=1.0, temp_soft_seq_max=1.0,
             seq_hard=True, sample=False,
             init_sd=0.01, beta_i=0.1, beta_f=10,
             track_freq=10, verbose=True, train=True,
             keep_first=False, pdb_idx=None, bkgs=None,
             L_start=None, track_steps=None,
             b1=0.9, b2=0.999,):
    
    ##########################################
    # initialize
    ##########################################
    graph_inputs = copy.deepcopy(graph_inputs)  # Don't want dict values to be modified!
    weights_list = [weights.get(x,1) for x in self.loss_label]
    if 'I' not in graph_inputs:  # no seed sequence passed
      graph_inputs['I'] = np.zeros(shape=(1,msa_num,L_start,20))
    graph_inputs['I'] += np.random.normal(0,init_sd,size=(1,msa_num,L_start,20))
    graph_inputs['loss_weights'] = np.array(weights_list)[None]
    graph_inputs['train'] = np.array([train])
    traj_loss = {label: [] for label in self.loss_label}
    branch_ws = []
    branch_ls = []
                   
    if bkgs is not None:
      graph_inputs['bkg'] = bkgs[L_start][None]
    best = {"loss":np.inf,"I":None}
    seq_curr = {'loss': np.inf}
    B_choice = 8
    track_step = {'step': []}
    mt,vt = 0,0
    n_aa = graph_inputs["I"].shape[1] * graph_inputs["I"].shape[2]
    print(f"loss weights: {graph_inputs['loss_weights']}")
    
    ##########################################  
    # optimize
    ##########################################
    for k in range(opt_iter):
      # softmax gumbel controls
      temp_soft_seq = temp_soft_seq_min + (temp_soft_seq_max - temp_soft_seq_min) * np.power(1 - k/opt_iter, temp_soft_seq_decay)
      graph_inputs['temp'] = np.array([temp_soft_seq])
      
      # permute input (for msa_design)
      if not keep_first:
        idx = np.random.permutation(np.arange(graph_inputs["I"].shape[1]))
        graph_inputs["I"] = graph_inputs["I"][:,idx]
        
      # ivan cs beta control
      graph_inputs['beta_ia'] = np.array([beta_i * np.power(beta_f/beta_i, k/(opt_iter-1))])
      graph_inputs['beta_cs'] = np.array([beta_i * np.power(beta_f/beta_i, k/(opt_iter-1))])
      
      #print('ContigSearch Beta', graph_inputs['beta_cs'])

      ##################
      # compute gradient and graph outputs
      ##################
      p = self.predict(graph_inputs)
      
      # add some trajectory stats
      loss_dict = dict(zip(self.loss_label,p['loss'][0]))
      for label in self.loss_label:
        traj_loss[label].append(loss_dict[label])

      w_sorted = np.sort(p['branch_weights'][0])[::-1]
      w_sorted_pad = np.zeros([30])
      w_sorted_pad[:w_sorted.shape[0]] = w_sorted
      branch_ws.append(w_sorted_pad)
      #print('Branch weights', w_sorted, w_sorted.shape)
      
      l_sorted = np.sort(p['branch_losses'][0][::-1])
      l_sorted_pad = np.zeros([30])
      l_sorted_pad[:l_sorted.shape[0]] = l_sorted
      branch_ls.append(l_sorted_pad)      
        
      # translate from idx0 to pdb_idx
      if 'con_ref_idx0' in p:
        p['con_ref_pdb_idx'] = [pdb_idx[idx0] for idx0 in p['con_ref_idx0'][0]]  # [('A', 1)]
        p['con_hal_pdb_idx'] = [('A', idx0+1) for idx0 in p['con_hal_idx0'][0]]
        
      ##################
      # save best result
      ##################
      if (np.sum(p['loss']) < np.sum(best["loss"])) or self.feat_drop > 0:  # or self.sample
        best = {"loss":p['loss'],
                "I":np.copy(graph_inputs["I"])
               }
        track_best = copy.copy(p)

      # gradient descent w/ decay
      if opt_method == "GD_decay":
        L = graph_inputs["I"].shape[-2]
        p['grad'] /= np.sqrt(np.square(p['grad']).sum((-1,-2),keepdims=True)) + 1e-8
        lr = opt_rate * np.sqrt(L) * np.power(1 - k/opt_iter, opt_decay)                     # why no np.sqrt(L) here? Chris and Basile did a lot of work without it.
        print('The decay lr is:', lr)
        graph_inputs["I"] -= lr * p['grad']

      # gradient scaling constant
      if opt_method == "GD_constant":
        L = graph_inputs["I"].shape[-2]
        p['grad'] /= np.sqrt(np.square(p['grad']).sum((-1,-2),keepdims=True)) + 1e-8
        lr = opt_rate * np.sqrt(L)
        graph_inputs["I"] -= lr * p['grad']

      # SGD
      if opt_method == "GD_SGD":
        L = graph_inputs["I"].shape[-2]
        lr = opt_rate                 # no need to scale by np.sqrt(L) because gradient is not normalized
        graph_inputs["I"] -= lr * p['grad']

      # ADAM optimizer w/ built-in decay
      if opt_method == "ADAM":
        mt_tmp = b1*mt + (1-b1)*p['grad']
        vt_tmp = b2*vt + (1-b2)*np.square(p['grad']).sum((-1,-2),keepdims=True)
        lr = opt_rate/np.sqrt(vt_tmp + 1e-8)
        graph_inputs["I"] -= lr * mt_tmp

      if opt_method == 'MCMC_biased':
        # Should only be used with single sequence design and argmax (not sampling)
        if k % 5000 == 0:
          B_choice *= 2
        seq_prop = {'loss': p['loss'][0,0],
                    'seq': np.eye(20)[graph_inputs['I'].argmax(-1)],  # one hot encode the sequence. Really only necesary for first step
                    'grad': grad
                   }
        
        # update tracker info
        track_cur['seq_prop'] = copy.copy(seq_prop)
        track_cur['seq_curr'] = copy.copy(seq_curr)
        track_cur['delta_loss'] = seq_prop['loss'] - seq_curr['loss']
        
        seq_new, seq_curr = MCMC_biased(seq_curr, seq_prop, B_choice, L_start, p_indel=0.0)
        
        # update inputs for next step
        graph_inputs['I'] = np.copy(seq_new)
        if bkgs is not None:
          graph_inputs['bkg'] = bkgs[graph_inputs['I'].shape[-2]][None]
      
      if track_steps is not None: track_freq = track_steps
      if verbose and (k+1) % track_freq == 0:
        print(f"{k+1} loss:{arr2str(p['loss'])}\n")
        if track_steps: track_step[k+1] = track_cur

    # compute output for best sequence
    print('computing the output of the best sequence')
    graph_inputs["I"] = best["I"]
    L_best = graph_inputs['I'].shape[-2]
    if bkgs is not None:
      graph_inputs['bkg'] = bkgs[L_best][None]
    graph_inputs['train'] = np.array([False])
    
    p = self.predict(graph_inputs)
    msa_logits = graph_inputs["I"][0]
    feat = p['feat'][0]
    loss = dict(zip(self.loss_label,p['loss'][0]))
    msa  = N_to_AA(msa_logits.argmax(-1))
    track_best['loss_nodrop'] = loss
    track_best['traj_loss'] = traj_loss
    track_best['branch_ws'] = branch_ws
    track_best['branch_losses'] = branch_ls

    return {"msa":msa, "loss":loss, "feat":feat,
            "msa_logits":msa_logits,
            'track_step': track_step, 'track_best': track_best}

  ###############################################################################
  def predict(self, graph_inputs):
    '''
    Essentially a wrapper to pass a feed_dict to the model and get
    back a dictionary of output tensors
    '''
    inputs_list = to_list(self.inputs_runtime_labels, graph_inputs)    
    if self.serial:
      preds = [[] for _ in range(len(self.model.outputs))]
      for model_weights in self.models:
        self.model.set_weights(model_weights)
        for n,o in enumerate(self.model.predict(inputs_list)):
          preds[n].append(o)
      graph_outputs =  to_dict(self.out_label, [np.mean(pred,0) for pred in preds])
    else:
      graph_outputs = to_dict(self.out_label, self.model.predict(inputs_list))
      
    graph_outputs["I"] = graph_inputs["I"]
    return graph_outputs
  ###############################################################################
    
##################################################################################
# process input features
##################################################################################
class MRF(Layer):
  def __init__(self, lam=4.5, lid=[0.3,18.0], uid=[1,0], use_entropy=False):
    super(MRF, self).__init__()
    self.lam = lam
    self.use_entropy = use_entropy
    self.lid, self.lid_scale = lid
    self.uid, self.uid_scale = uid

  def call(self, inputs):
    x = inputs[0]
    N,L,A = [tf.shape(x)[k] for k in range(3)]
    F = L*A

    with tf.name_scope('reweight'):
      if self.lid > 0 or self.uid < 1:
        id_len = tf.cast(L, tf.float32)
        id_mtx = tf.tensordot(x,x,[[1,2],[1,2]]) / id_len
        id_mask = []
        # downweight distant sequences
        if self.lid > 0: id_mask.append(tf.sigmoid((self.lid-id_mtx) * self.lid_scale))
        # downweight close sequences
        if self.uid < 1: id_mask.append(tf.sigmoid((id_mtx-self.uid) * self.uid_scale))
        weights = 1.0/(tf.reduce_sum(sum(id_mask),-1) + (self.uid == 1))
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
      cov_reg = cov + I_F * self.lam/tf.sqrt(tf.reduce_sum(weights))
      inv_cov = tf.linalg.inv(cov_reg + tf.random.uniform(tf.shape(cov_reg)) * 1e-8)

      x1 = tf.reshape(inv_cov,(L,A,L,A))
      x2 = tf.transpose(x1, [0,2,1,3])
      features = tf.reshape(x2, (L,L,A*A))

      # extract contacts
      x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * rm_diag
      x3_ap = tf.reduce_sum(x3,0)
      x4 = (x3_ap[None,:] * x3_ap[:,None]) / tf.reduce_sum(x3_ap)
      contacts = (x3 - x4) * rm_diag

      # combine 2D features
      feat_2D = tf.concat([features, contacts[:,:,None]], axis=-1)

    with tf.name_scope('1d_features'):
      # sequence
      x_i = tf.stop_gradient(x[0,:,:20])
      # pssm
      f_i = tf.reduce_sum(weights[:,None,None] * x, axis=0) / tf.reduce_sum(weights)
      # entropy
      if self.use_entropy:
        h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
      else:
        h_i = tf.zeros((L,1))
      # tile and combine 1D features
      feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
      feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
      feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    # combine 1D and 2D features
    feat = tf.concat([feat_1D_tile_A,feat_1D_tile_B,feat_2D],axis=-1)
    return tf.reshape(feat, [1,L,L,442+2*42])

class PSSM(Layer):
  # modified from MRF to only output tiled 1D features
  def __init__(self, diag=0.4, use_entropy=False):
    super(PSSM, self).__init__()
    self.diag = diag
    self.use_entropy = use_entropy
  def call(self, inputs):
    x,y = inputs
    _,_,L,A = [tf.shape(y)[k] for k in range(4)]
    with tf.name_scope('1d_features'):
      # sequence
      x_i = x[0,0,:,:20]
      # pssm
      f_i = y[0,0]
      # entropy
      if self.use_entropy:
        h_i = K.sum(-f_i * K.log(f_i + 1e-8), axis=-1, keepdims=True)
      else:
        h_i = tf.zeros((L,1))
      # tile and combined 1D features
      feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
      feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
      feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    with tf.name_scope('2d_features'):
      ic = self.diag * tf.eye(L*A)
      ic = tf.reshape(ic,(L,A,L,A))
      ic = tf.transpose(ic,(0,2,1,3))
      ic = tf.reshape(ic,(L,L,A*A))
      i0 = tf.zeros([L,L,1])
      feat_2D = tf.concat([ic,i0], axis=-1)

    feat = tf.concat([feat_1D_tile_A, feat_1D_tile_B, feat_2D],axis=-1)
    return tf.reshape(feat, [1,L,L,442+2*42])

def categorical(y_logits, temp=1.0, sample=False, hard=True):
  # ref: https://blog.evjang.com/2016/11/tutorial-categorical-variational.html

  def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)
  
  def gumbel_softmax_sample(logits): 
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temp,-1)
  
  def one_hot(x):
    y = tf.one_hot(tf.argmax(x,-1),tf.shape(x)[-1])  # argmax
    return tf.stop_gradient(y-x)+x                   # gradient bypass
  
  y_soft = tf.nn.softmax(y_logits/temp,-1)  
  y_soft = K.switch(sample, gumbel_softmax_sample(y_logits), y_soft)    
  y_hard = K.switch(hard, one_hot(y_soft), y_soft)
  return y_soft, y_hard

#############################################################################
# Layer for finding best contig placement
#############################################################################
class ContigSearch(Layer):
  '''
  Places contigs in predicted 6D geometry such that the weighted
  combination of cce and kl terms is minimized
  '''
  def __init__(self, ptn_geo, contigs, fix_N=False, fix_C=False, keep_order=True):
    super(ContigSearch, self).__init__()
    self.ptn_geo = ptn_geo
    self.contigs = contigs  # [[s1,e1], [s2,e2],...] 
    self.ncon = len(contigs)
    self.con_lens = np.array([y - x + 1 for x,y in contigs])
    self.fix_N = fix_N
    self.fix_C = fix_C
    self.keep_order = keep_order
    self.eps = 1e-8

    '''
    Pre-calculate the kernels for all contig interactions
    '''
    self.geo_ker = []
    self.ker_sizes = np.zeros((self.ncon, self.ncon), dtype=np.float32)
    for i in range(self.ncon):
      row = []
      for j in range(self.ncon):
        s1, e1 = contigs[i]
        s2, e2 = contigs[j]
        x = ptn_geo[s1:e1+1, s2:e2+1]
        x[...,[0,25,38,75,100,125]] = 0  # don't reward matches for cb > 20A bins or k=0 diag
        row.append(x)
        self.ker_sizes[i,j] = x.sum() / 6  # checked!: k=0 diag are 0                
      self.geo_ker.append(row)

  def call(self, inputs, depth=3, weight_cce=1., weight_kl=1.):
    ptn_hal, bkg, beta = inputs  # hacky, but keras layers can only have one positional arg!
    '''Note ptn_hal will be O_feat, a TENSOR, from trR forward pass'''
    
    def mk_mask(ptn_hal):
      with tf.name_scope("mk_mask") as scope:
        '''
        Boolean mask of valid starting indices for each contig in TERM
        Ensures contigs do not overlap with each other or ptn boundaries
        '''
        mask = tf.zeros([self.ncon, self.ncon, L, L], dtype=tf.float32)
        for i in range(self.ncon):
          for j in range(i+1, self.ncon):
            nl = self.con_lens[:i].sum()
            ni = self.con_lens[i]
            nm = self.con_lens[i+1:j].sum()
            nj = self.con_lens[j]
            nr = self.con_lens[j+1:].sum()

            # block is 1 when two contigs can begin at the i,j position and not clash
            if self.keep_order:
              L_block = L - nl - nj - nr + 1
              block = 1 - tf.linalg.band_part(tf.ones([L_block,L_block]), -1, ni+nm-1)
              if i==0 and self.fix_N: 
                mask_N = tf.concat([[1.0], tf.zeros([tf.shape(block)[0] - 1])], axis=0)
                block *= mask_N[:,None]
              if j==self.ncon-1 and self.fix_C:
                mask_C = tf.concat([tf.zeros([tf.shape(block)[1] - 1]), [1.0]], axis=0)
                block *= mask_C[None,:]
              block = tf.pad(block, [[nl, nj+nr-1], [nl, nj+nr-1]], "CONSTANT")

            else:   # still experimental
              block = 1 - tf.linalg.band_part(tf.ones([L-ni+1, L-nj+1]), nj-1, ni-1)
              block = tf.pad(block, [[0, ni-1], [0, nj-1]])
            
            # insert block slice into mask
            indices = tf.constant([[i,j]])
            updates = tf.cast(block[None], dtype=tf.float32)
            mask = tf.tensor_scatter_nd_add(mask, indices, updates)
      return mask + tf.transpose(mask, [1,0,3,2])

    def conv_contigs(ptn_hal):
      with tf.name_scope("con_contigs") as scope:
        '''
        Calc delta in total CCE and KL at each position (if it were the
        first aa of the contig). Does this by convolving the a kernel of
        contig-contig (intra or inter) over O_feat (for CCE) or bkg (for KL).
        '''

        ncon = len(self.contigs)
        cce_2D = tf.zeros([ncon, ncon, L, L])
        cce_1D = tf.zeros([ncon, L])
        kl_2D = tf.zeros([ncon, ncon, L, L])
        kl_1D = tf.zeros([ncon, L])                
        kl_ij = tf.math.reduce_sum(-ptn_hal * tf.math.log(ptn_hal/(bkg+self.eps)+self.eps), -1, keepdims=True) / 6  # batch,H,W,ch
        kl_ij *= 1 - tf.eye(L)[None,:,:,None]  # exclude diagonal
        kl_tot = tf.math.reduce_sum(kl_ij)

        for i in range(ncon):
          for j in range(i, ncon):
            # set up kernel
            ker = self.geo_ker[i][j]
            ker = tf.cast(ker, dtype=tf.float32)

            # conv for sum cce in kernel
            conv_cce_sum = tf.nn.conv2d(-tf.math.log(ptn_hal + self.eps), ker[...,None], 1, 'VALID') / 6  # this conv does NOT flip kernel
            if self.geo_ker[i][j].sum() < 0.5:  # catch case where all inter-residue dist > 20A
              conv_cce_sum = 1e6 * tf.ones(tf.shape(conv_cce_sum))  # only very large, so that it's a "second tier" pick
            conv_cce_sum = conv_cce_sum[0,:,:,0]

            # conv for sum kl in kernel
            ker_2D = tf.math.reduce_sum(ker, -1)[...,None,None] / 6  # H,W,ch_in,ch_out
            conv_kl_sum = tf.nn.conv2d(kl_ij, ker_2D, 1, 'VALID')
            conv_kl_sum = conv_kl_sum[0,:,:,0]  # okay if ker_sz is 0

            # score positions based on INTRA-contig geometry
            if i == j:
              def scatter_1D(conv, feat_1D, i):
                '''
                Place 1D features (intra-contig interactions, which
                show on diagonal of conv) into the appropriate tensor
                '''
                self_int = tf.linalg.diag_part(conv)
                x = tf.meshgrid(i, tf.range(tf.shape(conv)[0]))
                x = [tf.squeeze(el) for el in x]
                indices = tf.stack(x, -1)
                feat_1D = tf.tensor_scatter_nd_add(feat_1D, indices, self_int)
                return feat_1D
              cce_1D = scatter_1D(conv_cce_sum, cce_1D, i)
              kl_1D = scatter_1D(conv_kl_sum, kl_1D, i)

            # score positions based on INTER-contig geometry    
            else:
              def scatter_2D(conv, feat_2D, i, j):
                '''
                Place inter-contigs features into the appropriate tensor
                '''
                # scattering conv into cce is a bit tedious in tf...
                x = tf.meshgrid(i,
                                j,
                                tf.range(tf.shape(conv)[0]),
                                tf.range(tf.shape(conv)[1]),
                                indexing='ij')
                x = [tf.squeeze(el) for el in x]
                indices = tf.stack(x, -1)
                feat_2D = tf.tensor_scatter_nd_add(feat_2D, indices, conv)
                return feat_2D
              cce_2D = scatter_2D(conv_cce_sum, cce_2D, i, j)
              kl_2D = scatter_2D(conv_kl_sum, kl_2D, i, j)

        # make symmetric
        cce_2D += tf.transpose(cce_2D, (1,0,3,2))
        kl_2D += tf.transpose(kl_2D, (1,0,3,2))

        # set clashing region loss to +inf (CCE) or -inf (KL)
        mask = tf.cast(mk_mask(ptn_hal), dtype=tf.bool)  # true = no clashing. Let values through
        cce_2D = tf.where(mask, cce_2D, tf.fill([ncon,ncon,L,L], np.inf))  # does not do broadcasting well!
        kl_2D = tf.where(mask, kl_2D, tf.fill([ncon,ncon,L,L], -np.inf))  # does not do broadcasting well!

        # record size of intra (1D) and inter (2D) contig kernels
        ij_1D = tf.linalg.diag_part(self.ker_sizes)
        ij_1D = tf.tile(ij_1D[:,None], [1,L])
        ij_2D = (1. - tf.eye(self.ncon, dtype=tf.float32)) * tf.constant(self.ker_sizes, dtype=tf.float32)
        ij_2D = tf.tile(ij_2D[...,None,None], [1,1,L,L])
        ij_tot = tf.cast(L*(L-1), dtype=tf.float32)

      return cce_2D, cce_1D, kl_2D, kl_1D, kl_tot, ij_2D, ij_1D, ij_tot

    def mk_tree(depth=depth):
      with tf.name_scope("mk_tree") as scope:
        '''
        Find best positionings of contigs by adding top depth matches
        to existing contig clique at every layer
        '''
        # step 0: calc total loss when one contig pair is placed
        def combine_2D_and_1D(feat_1D, feat_2D):
          return 2*feat_2D + tf.tile(feat_1D[:,None,:,None], [1,ncon,1,L]) + tf.tile(feat_1D[None,:,None,:], [ncon,1,L,1])
        cce = combine_2D_and_1D(cce_1D, cce_2D)
        kl = combine_2D_and_1D(kl_1D, kl_2D)
        ij = combine_2D_and_1D(ij_1D, ij_2D)
        feat_1D_2D = tf.stack([cce, kl, ij], -1)
        loss_1D_2D, cce_mean_1D_2D, kl_mean_1D_2D = feat_2_loss(feat_1D_2D)

        # step 1: Find best depth**2 contig interactions
        # mask out the lower left triangle, as loss is symmetric in this regard. want to find unique interactions first
        mask_triu = tf.linalg.band_part(tf.ones([ncon,ncon]), 0, -1)  # 1 = let through value
        mask_triu = tf.cast(mask_triu, dtype=tf.bool)[...,None,None]
        mask_triu = tf.broadcast_to(mask_triu, tf.shape(cce))
        loss_unique = tf.where(mask_triu, tf.identity(loss_1D_2D), tf.fill([ncon,ncon,L,L], np.inf))  # does not do broadcasting well!

        # lowest loss positions
        _, args = tf.math.top_k(tf.reshape(-loss_unique, [-1]), depth**2)  # find args of the lowest depth**2 losses
        locs = tf.unravel_index(args, tf.shape(loss_unique))
        locs = tf.transpose(locs, (1,0))
        cons = locs[:, :2]
        idxs = locs[:, 2:]

        # start keeping track of cummulative values
        cce_cum_sum = tf.gather_nd(cce, locs)
        kl_cum_sum = tf.gather_nd(kl, locs)
        ij_cum_sum = tf.gather_nd(ij, locs)
        feat_cum_sum = tf.stack([cce_cum_sum, kl_cum_sum, ij_cum_sum], -1)  # branch, ch

        # stack cce, kl and ij features into "channels" for easy handling
        feat_1D = tf.stack([cce_1D, kl_1D, ij_1D], -1)  # ncon,L,ch
        feat_2D = tf.stack([cce_2D, kl_2D, ij_2D], -1)
        feat_2D = tf.transpose(feat_2D, [0,2,1,3,4])  # ncon,L,ncon,L,ch to make slicing easier (only need to specify ncon,idx to take slice)

        # step 2: Find next best contig placement given existing cliques
        for _ in range(len(self.contigs) - 2):
          # find change in features
          # -slice out "rows" from feat_2D
          outer_idxs = tf.stack([cons, idxs], -1)
          slices_2D = 2 * tf.gather_nd(feat_2D, outer_idxs)  # 2 is to correct asymmetry in what 2D features represent. branch,existing_nodes,ncon,L,ch

          # -add 1D features
          feat_delta = tf.math.reduce_sum(slices_2D, axis=1)
          feat_delta += feat_1D[None]  # branch, ncon, L, ch

          # add change in features to existing feature totals
          feat_updated = feat_cum_sum[:,None,None,:] + feat_delta  # branch,ncon,L,ch = branch,1,1,ch + branch,ncon,L,ch

          # calc loss as if contig were placed at every possible position
          loss, cce_mean, kl_mean = feat_2_loss(feat_updated)  # branch,ncon,L

          # find lowest loss contig positions for each branch
          n_branch = tf.shape(loss)[0]
          _, args = tf.math.top_k(tf.reshape(-loss, [n_branch, -1]), depth)  # branch,depth
          x = tf.unravel_index(tf.reshape(args, [-1]), tf.shape(loss)[-2:])  # only takes a 1D input...
          cons_to = tf.reshape(x[0], tf.shape(args))  # ...thuse we have to reshape here
          idxs_to = tf.reshape(x[1], tf.shape(args))  # branch, top depth hits

          # add new matches to existing cliques and expand number of branches
          def add_branches(branches, to_add):
            branches_new = tf.tile(branches[:,None,:], [1,depth,1])
            branches_new = tf.concat([branches_new, to_add[:,:,None]], axis=-1)  # branch, depth, new clique size
            existing_clique_size = tf.shape(idxs)[-1]
            branches_new = tf.reshape(branches_new, [-1, existing_clique_size + 1])  # branch, new clique size.
            return branches_new
          cons = add_branches(cons, cons_to)
          idxs = add_branches(idxs, idxs_to)

          # update feat_cum_sum
          a = tf.range(n_branch)[:,None]
          a = tf.broadcast_to(a, tf.shape(cons_to))
          feat_delta_selected = tf.gather_nd(feat_delta, tf.stack([a, cons_to, idxs_to], -1))
          feat_cum_sum = feat_cum_sum[:,None,:] + feat_delta_selected  # branch, depth, ch
          feat_cum_sum = tf.reshape(feat_cum_sum, [-1, 3])  # branch, ch

        # calc final loss
        final_loss, final_cce_mean, final_kl_mean = feat_2_loss(feat_cum_sum)
      return final_loss, final_cce_mean, final_kl_mean, cons, idxs

    def feat_2_loss(feat_cum_sum):
      '''
      Covert from individual features to total loss
      feat: branch, ch (ch= cce_cum_sum, kl_cum_sum, ij_cum_sum)
      '''
      ch_cce, ch_kl, ch_ij = split_channels(feat_cum_sum)
      cce_mean_unweighted = ch_cce / ch_ij
      kl_mean_unweighted = (kl_tot - ch_kl) / (ij_tot - ch_ij)
      loss = weight_cce * cce_mean_unweighted + weight_kl * kl_mean_unweighted  # branch,ncon,L
      return loss, cce_mean_unweighted, kl_mean_unweighted

    def split_channels(feat):
      ch_cce = feat[...,0]
      ch_kl  = feat[...,1]
      ch_ij  = feat[...,2]
      return ch_cce, ch_kl, ch_ij
    
    def mk_idxs(con, idx):
      '''
      Convert from start of contig index to an array of indices that cover each
      residue of the the contig. Can be used with gather_nd to gather the correct
      ranges of the input pdb.

      Ex:
      con = np.array([
        [0,1,2],
        [2,0,1],
        [0,2,1],
        [2,1,0],
        [1,0,2]
      ])

      idx = np.array([
        [22,11,33],
        [14,7,29],
        [2,9,27],
        [13,27,33],
        [34,14,7]
      ])

      contigs = np.array([[1,3], [4,7], [9,13]])
      con_lens = np.array([y - x + 1 for x,y in contigs])

      array([[22, 23, 24, 11, 12, 13, 14, 33, 34, 35, 36, 37],
             [ 7,  8,  9, 29, 30, 31, 32, 14, 15, 16, 17, 18],
             [ 2,  3,  4, 27, 28, 29, 30,  9, 10, 11, 12, 13],
             [33, 34, 35, 27, 28, 29, 30, 13, 14, 15, 16, 17],
             [14, 15, 16, 34, 35, 36, 37,  7,  8,  9, 10, 11]])
      '''
      # sort the start idx by ascending contig order
      n_branch = tf.shape(con)[0]
      col = tf.argsort(con)
      row = tf.range(n_branch)[:,None]
      row = tf.broadcast_to(row, con.shape)
      reorder = tf.stack([row, col], -1)
      s_ordered = tf.gather_nd(idx, reorder)

      # for each row, add a range of contig length, starting at s_ordered
      ls = []
      for i in range(self.ncon):
        range_ = tf.broadcast_to(tf.range(self.con_lens[i], dtype=tf.int32), (n_branch, self.con_lens[i]))
        start = tf.broadcast_to(s_ordered[:,i][:,None], (n_branch, self.con_lens[i]))
        ls.append(start + range_)

      return tf.concat(ls, -1)
    
    def tf_unique_2d(x):
      '''
      Finds unique rows in 2d tensor
      from: https://stackoverflow.com/questions/51487990/find-unique-values-in-a-2d-tensor-using-tensorflow
      '''
      x_shape = tf.shape(x)  # (3,2)
      x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
      x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

      x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
      x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
      cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
      cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
      cond_shape = tf.shape(cond)
      cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
      cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

      # CREATING RANGE TENSOR
      r = tf.range(x_shape[0])
      r = tf.add(tf.tile(r, [x_shape[0]]), 1)
      r = tf.reshape(r, [x_shape[0], x_shape[0]])

      # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
      f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
      f2 = tf.ones(cond_shape, tf.int32)
      cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

      # multiply range with new int boolean mask
      r_cond_mul = tf.multiply(r, cond_cast2)
      r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
      r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
      r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

      # get actual values from unique indexes
      op = tf.gather(x, r_cond_mul4)

      # make boolean array for unique rows
      idx_uniq = r_cond_mul4
      mask_uniq = tf.scatter_nd(idx_uniq[:,None], tf.ones(tf.shape(idx_uniq)), shape=[tf.shape(x)[0]])
      mask_uniq = tf.cast(mask_uniq, dtype=tf.bool)
      
      return op, mask_uniq
    
    ####################################################################################
    # finding the best placements of the contigs is so easy with abstraction!
    ####################################################################################
    L = tf.shape(ptn_hal)[1]
    ncon = len(self.contigs)
    cce_2D, cce_1D, kl_2D, kl_1D, kl_tot, ij_2D, ij_1D, ij_tot = conv_contigs(ptn_hal)
    final_loss, final_cce_mean_unweighted, final_kl_mean_unweighted, cons, idxs = mk_tree()
    
    # find best placement. Can upgrade to a softmax version in the future
    #best_arg = tf.math.argmin(final_loss)
    #best_con = cons[best_arg]
    #best_idx = idxs[best_arg]
    #return best_cce_unweighted, best_kl_unweighted, mk_idxs(best_con, best_idx)
    
    # upgrading to a softmax version
    # mask for entries with inf loss (clashes)
    mask_inf = tf.math.equal(final_loss, np.inf)
    
    # mask for unique entries
    unique_rows, mask_uniq = tf_unique_2d(tf.sort(idxs))
    
    # combine masks
    mask_rows = tf.math.logical_and(~mask_inf, mask_uniq)
  
    # only score "boltzman energies" of unique contigs that don't clash
    final_loss = final_loss[mask_rows]  # this is a weighted loss
    final_cce_mean_unweighted = final_cce_mean_unweighted[mask_rows]
    final_kl_mean_unweighted = final_kl_mean_unweighted[mask_rows]
    
    # weight cce and kl scores by the corresponding loss
    branch_weights = tf.nn.softmax(-final_loss * beta)
    cce_unweighted_super = tf.math.reduce_sum(branch_weights * final_cce_mean_unweighted)
    kl_unweighted_super  = tf.math.reduce_sum(branch_weights * final_kl_mean_unweighted)
    
    # find con_idxs0 for all/best branches
    branch_con_idxs0 = mk_idxs(cons, idxs)
    branch_con_idxs0 = branch_con_idxs0[mask_rows]
    best_con_idxs0 = branch_con_idxs0[tf.math.argmin(final_loss)]
    
    return cce_unweighted_super, kl_unweighted_super, best_con_idxs0, branch_weights, final_loss, branch_con_idxs0
  
  
#######################################
# Ivan contig search
#######################################
def probe_bsite_tf_frag(pred, beta, L, bsite, idx, bin_width='15_deg'):
    
    '''
    -------------------------------------------------------------------------
    - pred[?,L,L,130]    : trRosetta predictions (tf.tensor)
    - bsite[m,m,130]     : 1-hot encoded binding site, 6D coords (np.array)
    - idx[m]             : indices of bs residues (np.array, int)
    - beta               : inverse temperature, a scalar (float)
    - L
    - bin_width          : do the angle bins span 10 or 15 degrees? Allow compatibility
                           with old and new trRosetta versions
     
     If bin_width is '15_deg', these changes apply:
     - pred[?,L,L,138]   : trR predictions stacked [p_theta, p_phi, p_dist, p_omega, p_theta_T, p_phi_T]
     - bsite[m,m,138]    : 1-hot encoded binding site, 6D coords, stacked [p_theta, p_phi, p_dist, p_omega, p_theta_T, p_phi_T]
    -------------------------------------------------------------------------
    210104 - dt lightly modified to handle inputs of 15deg trR model
    '''

    if bin_width == '10_deg':
      # extend 2D arrays so that each pixel stores the full set coordinates (6D)
      pred = tf.concat([pred,tf.transpose(pred[:,:,:,74:],[0,2,1,3])], axis=-1)
      bsite_np = np.concatenate([bsite,np.transpose(bsite[:,:,74:],[1,0,2])], axis=-1)
    elif bin_width == '15_deg':
      # features are already symmetric
      bsite_np = np.array(bsite)
    
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
    if bin_width == '10_deg':
      pad = np.array([0]*36+[1] + [0]*36+[1] + [0]*36+[1] + [0]*18+[1] + [0]*36+[1] + [0]*18+[1]).reshape((1,1,1,186))
    elif bin_width == '15_deg':
      pad = np.array([1]+[0]*24 + [1]+[0]*12 + [1]+[0]*36 + [1]+[0]*24 + [1]+[0]*24 + [1]+[0]*12).reshape((1,1,1,138))
    pad1 = tf.constant(np.tile(pad,(1,L,max_pad,1)),dtype=tf.float32)  #can probably pad with zeros
    pad2 = tf.constant(np.tile(pad,(1,max_pad,L+2*max_pad,1)),dtype=tf.float32)
    
    pred_pad = tf.concat([pad2,tf.concat([pad1,pred,pad1],axis=2),pad2],axis=1)
    
    pred_log = tf.math.log(pred_pad+1e-8)
    conv_ = lambda p : tf.nn.conv2d(pred_log,
                                    bs_c6d_ij[hash_(p)],
                                    strides=[1,1,1,1],
                                    padding="SAME")[:,max_pad:-max_pad,max_pad:-max_pad,0]

    # satisfaction score
    CCE_i = [-tf1.diag_part(conv_([i,i])[0])[tf.newaxis,:] for i in range(n)]
    
    penalty = {k: signal.convolve2d(np.eye(L), np.full(bs.shape[0:2],1), boundary='wrap', mode='same') #/np.max(bs.shape[0:2])  #can be converted to tf conv
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