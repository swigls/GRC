
"""
This module contains the layer base class :class:`LayerBase`,
and many canonical basic layers.
"""

from __future__ import print_function

import tensorflow as tf
from returnn.util.basic import NotSpecified
from returnn.tf.util.data import Data

## ADDED 
from .basic import _ConcatInputLayer, LayerBase, concat_sources, get_concat_sources_data_template, ConvLayer

##################################### ADDED #########################################
class MonotonicHardAttention2Layer(LayerBase):
  layer_class = "monotonic_hard_attention_2"

  # noinspection PyShadowingBuiltins
  def __init__(self, sources, energy_factor=None, sigmoid_noise=1.0, seed=None,\
               chunk_size=1, test_same_as_train=False,
               use_time_mask=None, train_cumcalc_mode="recursive", **kwargs):
    assert sources
    super(MonotonicHardAttention2Layer, self).__init__(sources=sources, **kwargs)
    from returnn.tf.util.basic import where_bc
    energy_data = concat_sources([self.sources[0]]) # (enc-T,B,H), not (B,enc-T,H) 
    assert energy_data.dtype.startswith("float")
    previous_attention_data = concat_sources([self.sources[1]]) #(enc-T,B,H)
    axis = 0 #energy_data.batch_ndim - 1
    energy = energy_data.placeholder #(enc-T,B,H)

    chunk_energy = None
    if chunk_size is not None and isinstance(chunk_size,(int,float)) and chunk_size > 1:
      chunk_size = int(chunk_size)
      chunk_energy_data = concat_sources([self.sources[2]]) #if chunk_size > 1 else None #(enc-T,B,H)
      chunk_energy = chunk_energy_data.placeholder 
    orig_time_axis = self._get_axis_to_reduce(input_data=energy_data, axis="T", exception_prefix=self)
    if orig_time_axis==1: # this case, transpose placeholder (B,enc-T,H)->(enc-T,B,H)
      print("original time axis of energy was 1, and is changed with 0-th axis! (to make (enc-T,B,H))")
      energy = tf.transpose(energy, perm=(1,0,2)) 
      if chunk_energy is not None:    
        print("  =>(did same for chunk_energy)")
        chunk_energy = tf.transpose(chunk_energy, perm=(1,0,2))     

    previous_attention = previous_attention_data.placeholder #(enc-T,B,H)
    orig_time_axis_prevatt = self._get_axis_to_reduce(input_data=previous_attention_data, axis="T", exception_prefix=self)
    if orig_time_axis_prevatt==1: # this case, transpose placeholder (B,enc-T,H)->(enc-T,B,H)
      print("original time axis of previous_attention was 1, and is changed with 0-th axis! (to make (enc-T,B,H))")
      previous_attention = tf.transpose(previous_attention, perm=(1,0,2)) 

    energy_shape = tf.shape(energy) #shape is (enc-T,B,H)
    init_ones = tf.ones([1, energy_shape[1], energy_shape[2]], dtype=energy.dtype) #(1,B,H)
    init_zeros = tf.zeros([energy_shape[0]-1, energy_shape[1], energy_shape[2]], dtype=energy.dtype) #(enc-T - 1,B,H)
    init_attention = tf.concat([init_ones, init_zeros], axis=axis)
    previous_attention = tf.cond(
      tf.equal(tf.reduce_sum(tf.abs(previous_attention)), tf.constant(0., dtype=previous_attention.dtype)),
      true_fn = lambda: init_attention,
      false_fn = lambda: previous_attention,
    )

    from returnn.tf.util.basic import check_shape_equal
    assert energy_data.have_time_axis()
    assert previous_attention_data.have_time_axis()
    # if the time-axis is static, we can skip the masking
    if use_time_mask is None:
      use_time_mask = energy_data.is_axis_dynamic(orig_time_axis)
    if use_time_mask:
      assert energy_data.is_axis_dynamic(orig_time_axis), "%s: use_time_mask True, dyn time axis expected" % self
      energy_mask = energy_data.get_sequence_mask_broadcast(axis=orig_time_axis)
      if orig_time_axis==1: # (B,enc-T,H)->(enc-T,B,H)
        energy_mask = tf.transpose(energy_mask, perm=(1,0,2)) 
      energy = where_bc(energy_mask, energy, float("-inf"), name="energy_masked")
      if chunk_energy is not None:
       chunk_energy = where_bc(energy_mask, chunk_energy, float("-inf"), name="chunk_energy_masked")
    if energy_factor:
      energy = tf.multiply(energy, energy_factor, name="energy_scaled")
      if chunk_energy is not None:
        chunk_energy = tf.multiply(chunk_energy, energy_factor, name="chunk_energy_scaled")

    ### main part (https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/parts/rnns/attention_wrapper.py)
    # Add sigmoid noise only at training
    network = self.sources[0].network
    score = energy #(enc-T,B,H)
    if network.train_flag is not False:
      print("----------------------------------------------------------")
      print("---------- NOW TRAIN TIME !!!!!!(sigmoide noise add)------")
      print("----------------------------------------------------------")
      if sigmoid_noise > 0:
        noise = tf.random.normal(tf.shape(score), dtype=score.dtype, seed=seed)
      score += sigmoid_noise * noise
    # Calculate p_choose_i
    if (network.train_flag is not False or test_same_as_train):
      print("----------------------------------------------------------")
      print("---------- NOW TRAIN TIME !!!!!!(p=sigmoid(score))--------")
      print("----------------------------------------------------------")
      p_choose_i = tf.sigmoid(score) #(enc-T,B,H)
    else:
      print("----------------------------------------------------------")
      print("---------- NOW TEST TIME !!!!!!(p=1(score>0))------------")
      print("----------------------------------------------------------")
      if True:
        p_choose_i = tf.cast(score > 0, score.dtype) #(enc-T,B,H)
      else: #sampling (not to be used)
        p_choose_i = tf.sigmoid(1.*score) #(enc-T,B,H)
        z = tf.random.uniform(tf.shape(score), dtype=score.dtype, seed=seed) #(enc-T,B,H)
        p_choose_i = tf.cast(p_choose_i > z, score.dtype)
    # Calculate weights
    if (network.train_flag is not False or test_same_as_train) and train_cumcalc_mode=="recursive":
      assert False, "Recursive mode is not implemented yet."
      print("----------------------------------------------------------")
      print("---------------- NOW TRAIN TIME !!!!!!(recursive)---------")
      print("----------------------------------------------------------")
      # Use .shape[0].value when it's not None, or fall back on symbolic shape
      batch_size = p_choose_i.shape[1].value or tf.shape(p_choose_i)[1]
      num_heads = p_choose_i.shape[2].value or tf.shape(p_choose_i)[2]
      # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
      shifted_1mp_choose_i = tf.concat(
        [tf.ones((1, batch_size, num_heads)), 1 - p_choose_i[:-1,:,:]], axis=0
      ) #(B,H,enc-T)
      # Compute attention distribution recursively as
      # q[i] = (1 - p_choose_i[i])*q[i - 1] + previous_attention[i]
      # attention[i] = p_choose_i[i]*q[i]
      weights = p_choose_i * tf.transpose(
        tf.scan(
          # Need to use reshape to remind TF of the shape between loop
          # iterations
          lambda x, yz: tf.reshape(yz[0] * x + yz[1], (batch_size,num_heads)),
          # Loop variables yz[0] and yz[1]
          [
            # (enc-T,B,H)
            tf.transpose(shifted_1mp_choose_i, perm=(0,1,2)),
            tf.transpose(previous_attention, perm=(0,1,2))
          ],
          # Initial value of x is just zeros
          tf.zeros((batch_size,num_heads)), #(B,H)
          swap_memory=True,
          parallel_iterations=1,
        ),
        # (enc-T,B,H)
        perm=(0,1,2)
      )
    elif (network.train_flag is not False or test_same_as_train) and train_cumcalc_mode=="parallel":
      print("----------------------------------------------------------")
      print("---------------- NOW TRAIN TIME !!!!!!(parallel)----------")
      print("----------------------------------------------------------")
      def safe_cumprod(x, *args, **kwargs):
        #with tf.name_scope(None, "SafeCumprod", [x]):
        with tf.name_scope("SafeCumprod"):
          x = tf.convert_to_tensor(x, name="x")
          import numpy as np
          tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
          return tf.exp(
                   tf.cumsum(
                     tf.log(tf.clip_by_value(x, tiny, 1)), *args, **kwargs
                   )
                 )
      # safe_cumprod computes cumprod in logspace with numeric checks
      cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, axis=axis, exclusive=True) #(enc-T,B,H) #(45,1,132)
      # Compute recurrence relation solution
      weights = p_choose_i * cumprod_1mp_choose_i * tf.cumsum(
        previous_attention /
        # Clip cumprod_1mp to avoid divide-by-zero
        tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.),
        axis=axis
      ) #(enc-T,B,H)
    elif (network.train_flag is not False or test_same_as_train):
      assert False, "train_cumcalc_mode must be in [\"recuresive\",\"parallel\"]"
    else:
      print("----------------------------------------------------------")
      print("---------------- NOW TEST TIME !!!!!!!--------------------")
      print("----------------------------------------------------------")
      ####### ORIG(openseq2seq) ##########
      # p_choose_i          : [0,0,1,1,1,1,0,0]
      # tf.cumsum(prev_att) : [0,0,1,1,1,1,1,1]
      # 1-p_choose_i        : [1,1,0,0,0,0,1,1]
      # tf.cumprod('')      : [1,1,1,0,0,0,0,0]
      #weights = p_choose_i * tf.cumsum(previous_attention, axis=axis) *\
      #          tf.cumprod(1 - p_choose_i, axis=axis, exclusive=True)  #(enc-T,B,H)
      ######## ADDED ########
      prev_att_existance = tf.cast(previous_attention > 0., dtype=tf.float32) #e.g. [0, 0.1, 0.2, 0.8, 0] => [0, 1, 1, 1, 0]
      reverse_filter = tf.cumsum(prev_att_existance, axis=axis, exclusive=True, reverse=True) #e.g. [0, 1, 1, 1, 0] => [3, 2, 1, 0, 0]
      reverse_filter_existance = tf.cast(reverse_filter > 0., dtype=tf.float32) #e.g. [3, 2, 1, 0, 0] => [1, 1, 1, 0, 0]
      filter_existance = 1 - reverse_filter_existance #e.g. [1, 1, 1, 0, 0] => [0, 0, 0, 1, 1]
      previous_hard_attention = prev_att_existance * filter_existance #e.g. [0, 1, 1, 1, 0] * [0, 0, 0, 1, 1] = [0, 0, 0, 1, 0]
      # p_choose_i          : [1,0,0,0,1,0,1,0]
      # prev_attention      : [0,0,1,0,0,0,0,0]
      # p_c *tf.cumsum('')  : [0,0,0,0,1,0,1,0]
      # tf.cumprod(1-(''))  : [1,1,1,1,1,0,0,0]
      # weights             : [0,0,0,0,1,0,0,0]
      cs_hard_attention = p_choose_i * tf.cumsum(previous_hard_attention, axis=axis) #(enc-T,B,H)
      weights = cs_hard_attention * tf.cumprod(1-cs_hard_attention, axis=axis, exclusive=True) #(enc-T,B,H)
      ##############################
    if isinstance(chunk_size, (int,float)) and chunk_size > 1:
      alpha = weights #(enc-T,B,H), only one t_i among enc-T is 1, others are 0.
      alpha_shape = tf.shape(alpha)
      if (network.train_flag is not False or test_same_as_train):
        def moving_sum(x, b, f): #x:(T,B,C)
          x_shape = tf.shape(x)
          x = tf.transpose(x,perm=(1,0,2)) #(T,B,C)->(B,T,C)
          I = tf.expand_dims(tf.eye(x_shape[2]), axis=0) #(1,C,C), no operation applied on head dimension
          filt_half = max(b,f) - 1 #assume b,f are not tensors
          filt = tf.concat(
            [tf.zeros([filt_half-(b-1), 1, 1], dtype=x.dtype),
            tf.ones(  [(b-1)+(1)+(f-1), 1, 1], dtype=x.dtype),
            tf.zeros( [filt_half-(f-1), 1, 1], dtype=x.dtype)],
            axis=0,
          ) 
          W = I * filt #(2*max(b,f)-1, C, C) 
          return tf.transpose(
            tf.nn.conv1d(x, W, stride=1, padding="SAME"), #(B,T,C)
            perm=(1,0,2) #(B,T,C)->(T,B,C)
          ) #zero-padding is enough, assuming that exp(u) comes in as input. (exp(-inf)==0) 
        exp_u = tf.exp(chunk_energy) #(enc-T,B,H)
        beta = exp_u * moving_sum(alpha / (moving_sum(exp_u, chunk_size, 1) + 1e-6), 1, chunk_size) #(enc-T,B,H)
      else:
        t = tf.argmax(alpha, axis=0) #(B,H)
        chunk_energy_mask = tf.logical_or(
          tf.sequence_mask(t+1-chunk_size, maxlen=alpha_shape[0], name='chunk_energy_mask_pre'), #(B,H,enc-T), bool  
          tf.logical_not(tf.sequence_mask(t+1, maxlen=alpha_shape[0], name='chunk_energy_mask_post')) #(B,H,enc-T), bool  
        )
        chunk_energy_mask = tf.where(
          tf.transpose(chunk_energy_mask, perm=(2,0,1)), #(B,H,enc-T) => (enc-T,B,H) 
          x=tf.ones(alpha_shape, dtype=tf.float32) * float('-inf'),
          y=tf.zeros(alpha_shape, dtype=tf.float32),
        )
        # softmax over (t_i-chunk_size+1,t_i)
        chunk_energy += chunk_energy_mask #(enc-T,B,H)
        beta = tf.where(
          tf.ones_like(alpha) * tf.reduce_sum(tf.abs(alpha), axis=0, keepdims=True) > 0., #(enc-T,B,H)
          x=tf.nn.softmax(chunk_energy, axis=0),
          y=tf.zeros_like(chunk_energy)
        )
      weights = beta
    ############################################
    if orig_time_axis==1: # this case, transpose placeholder (enc-T,B,H)->(B,enc-T,H)
      weights = tf.transpose(weights, perm=(1,0,2))     
    weights = tf.reshape(weights, [tf.shape(weights)[0], tf.shape(weights)[1], 1]) 
    self.output.placeholder = weights #(enc-T,B,H)
 
  @classmethod
  def get_out_data_from_opts(cls, n_out=NotSpecified, out_type=None, sources=(), **kwargs):
    """
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out_type_ = {}
    if sources and any(sources):
      out_type_.update(Data.get_common_data([s.output for s in sources if s]).get_kwargs())
    if n_out is not NotSpecified:
      out_type_["dim"] = n_out
    out_type_["name"] = "%s_output" % kwargs["name"]
    if out_type:
      if isinstance(out_type, dict):
        out_type_.update(out_type)
      elif callable(out_type):
        out_type_ = out_type  # just overwrite
      else:
        raise TypeError("unexpected type of out_type %r" % (out_type,))
    ######## ADDED ############
    #out_type_["batch_dim_axis"] = 0
    #out_type_["feature_dim_axis"] = 1
    #out_type_["time_dim_axis"] = 2
    ##################################
    return super(MonotonicHardAttention2Layer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type_, sources=sources, **kwargs)

  @classmethod
  def _get_axis_to_reduce(cls, input_data, axis, exception_prefix):
    """
    :param str|None axis:
    :param str|object exception_prefix:
    :rtype: int
    """
    if axis is None:
      assert input_data.have_time_axis(), "%s: requires that the input has a time dim" % exception_prefix
      axis = input_data.time_dim_axis
    else:
      axis = input_data.get_axis_from_description(axis, allow_int=False)
    return axis
######################################################################################
############### ADDED ######################
class GatedRecurrentContextLayer(LayerBase):
  layer_class = "gated_recurrent_context"

  # noinspection PyShadowingBuiltins
  def __init__(self, sources, seed=None, use_time_mask=None, \
               infer_threshold=None, first_reset_value=1., exp_energy_cumsum=False, sigmoid_energy_cumprod=False,
               **kwargs):
    assert sources
    super(GatedRecurrentContextLayer, self).__init__(sources=sources, **kwargs)
    from TFUtil import where_bc
    energy_data = concat_sources([self.sources[0]]) # (enc-T,B,H), not (B,enc-T,H) 
    assert energy_data.dtype.startswith("float")
    energy = energy_data.placeholder #(enc-T,B,H)
    axis = 0 #energy_data.batch_ndim - 1

    orig_time_axis = self._get_axis_to_reduce(input_data=energy_data, axis="T", exception_prefix=self)
    if orig_time_axis==1: # this case, transpose placeholder (B,enc-T,H)->(enc-T,B,H)
      print("original time axis of energy was 1, and is changed with 0-th axis! (to make (enc-T,B,H))")
      energy = tf.transpose(energy, perm=(1,0,2)) 

    energy_shape = tf.shape(energy) #shape is (enc-T,B,H)
    
    assert energy_data.have_time_axis()
    # if the time-axis is static, we can skip the masking
    if use_time_mask is None:
      use_time_mask = energy_data.is_axis_dynamic(orig_time_axis)
    if use_time_mask:
      assert energy_data.is_axis_dynamic(orig_time_axis), "%s: use_time_mask True, dyn time axis expected" % self
      energy_mask = energy_data.get_sequence_mask_broadcast(axis=orig_time_axis)
      if orig_time_axis==1: # (B,enc-T,H)->(enc-T,B,H)
        energy_mask = tf.transpose(energy_mask, perm=(1,0,2))
      energy = where_bc(energy_mask, energy, float("-inf"), name="energy_masked")
    ### Attention 
    # Add sigmoid noise only at training
    network = self.sources[0].network
    #(enc-T,B,H)
    def safe_cumprod(x, *args, **kwargs):
      #with tf.name_scope(None, "SafeCumprod", [x]):
      with tf.name_scope("SafeCumprod"):
        x = tf.convert_to_tensor(x, name="x")
        import numpy as np
        tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
        return tf.exp(
                 tf.cumsum(
                   tf.math.log(tf.clip_by_value(x, tiny, 1)), *args, **kwargs
                 )
               )
    if exp_energy_cumsum and sigmoid_energy_cumprod:
      assert False, "Use only 1 among exp_energy_cumsum or sigmoid_energy_cumprod."
    elif sigmoid_energy_cumprod:
      sigmoid_energy = tf.sigmoid(energy) #(enc-T,B,H)
      reset = safe_cumprod(sigmoid_energy) #(enc-T,B,H)
    elif exp_energy_cumsum:
      exp_energy = tf.exp(-energy)
      exp_energy_accum = tf.cumsum(exp_energy, axis=0 ) #(enc-T,B,H), increasing as time goes
      reset = 1. / (1. + exp_energy_accum) #(enc-T,B,H), 0~1, decreasing from 1 to 0 as time goes
    else:
      reset = tf.sigmoid(energy) #(enc-T,B,H), 0~1
    # 
    def substitute(x, time_pads=[0,0], value=0.):
      T = tf.shape(x)[0]
      x_left = tf.fill([time_pads[0],energy_shape[1],energy_shape[2]], value)
      x_middle = x[time_pads[0]:T-time_pads[1],:,:]
      x_right = tf.fill([time_pads[1],energy_shape[1],energy_shape[2]], value)
      return tf.concat([x_left, x_middle, x_right], axis=axis)
    if first_reset_value is not None:
      reset = substitute(reset, time_pads=[1,0], value=first_reset_value) 
    #
    if ((network.train_flag is None or network.train_flag is False) and infer_threshold is not None):
      print("----------------------------------------------------------")
      print("--------------------INFER_simple_thresholding-------------")
      print("----------------------------------------------------------")
      low_threshold_point = get_endpoint_compare_to(reset, infer_threshold, "l", "first")
      before_low_threshold = tf.cumsum(low_threshold_point, axis=0, reverse=True) 
      reset = before_low_threshold*reset 
    # safe_cumprod computes cumprod in logspace with numeric checks
    cumprod_1mreset = safe_cumprod(1 - reset, axis=axis, exclusive=True, reverse=True) #(enc-T,B,H) #(45,1,132)
    # Compute recurrence relation solution
    weights = reset * cumprod_1mreset
    ###
    if orig_time_axis==1: # this case, transpose placeholder (enc-T,B,H)->(B,enc-T,H)
      weights = tf.transpose(weights, perm=(1,0,2))     
    weights = tf.reshape(weights, [tf.shape(weights)[0], tf.shape(weights)[1], 1])
    self.output.placeholder = weights #(enc-T,B,H)
 
  @classmethod
  def get_out_data_from_opts(cls, n_out=NotSpecified, out_type=None, sources=(), **kwargs):
    """
    :param int|None|NotSpecified n_out:
    :param dict[str]|None out_type:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out_type_ = {}
    if sources and any(sources):
      out_type_.update(Data.get_common_data([s.output for s in sources if s]).get_kwargs())
    if n_out is not NotSpecified:
      out_type_["dim"] = n_out
    out_type_["name"] = "%s_output" % kwargs["name"]
    if out_type:
      if isinstance(out_type, dict):
        out_type_.update(out_type)
      elif callable(out_type):
        out_type_ = out_type  # just overwrite
      else:
        raise TypeError("unexpected type of out_type %r" % (out_type,))
    ######## ADDED ############
    #out_type_["batch_dim_axis"] = 0
    #out_type_["feature_dim_axis"] = 1
    #out_type_["time_dim_axis"] = 2
    ###########################
    return super(GatedRecurrentContextLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type_, sources=sources, **kwargs)
  @classmethod
  def _get_axis_to_reduce(cls, input_data, axis, exception_prefix):
    """
    :param str|None axis:
    :param str|object exception_prefix:
    :rtype: int
    """
    if axis is None:
      assert input_data.have_time_axis(), "%s: requires that the input has a time dim" % exception_prefix
      axis = input_data.time_dim_axis
    else:
      axis = input_data.get_axis_from_description(axis, allow_int=False)
    return axis
######################################################################################

############### ADDED ########################
### Variant of class::ReinterpretLayer in TFNetworkLayer.py
class ReinterpretAndCropLayer(_ConcatInputLayer):
  """
  Acts like the :class:`CopyLayer` but reinterprets the role of some axes or data.
  """
  layer_class = "reinterpret_and_crop"

  # noinspection PyUnusedLocal
  def __init__(self, switch_axes=None, size_base=None, set_axes=None,
               enforce_batch_major=False, enforce_time_major=False,
               set_sparse=None, set_sparse_dim=NotSpecified, increase_sparse_dim=None,
               **kwargs):
    """
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base:
    :param dict[str,int|str] set_axes: the key is "B","T","F", value is via :func:`Data.get_axis_from_description`
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param bool|None set_sparse: if bool, set sparse value to this
    :param int|None|NotSpecified set_sparse_dim: set sparse dim to this. assumes that it is sparse
    :param int|None increase_sparse_dim: add this to the dim. assumes that it is sparse
    """
    super(ReinterpretAndCropLayer, self).__init__(**kwargs)
    self.size_base = size_base
    ### ADDED ###
    #self.output.placeholder = self.input_data.placeholder
    x_data = concat_sources([self.sources[0]])
    base_data = concat_sources([self.size_base])
    x = x_data.placeholder
    base = base_data.placeholder
    base_time_axis = base_data.time_dim_axis #0 or 1 
    self.output.placeholder = x[:,:tf.shape(base)[base_time_axis],:]
    ####################
    if len(self.sources) == 1:
      self.output_loss = self.sources[0].output_loss
      if not self.dropout:
        self.output_before_activation = self.sources[0].output_before_activation
    for src in self.sources:
      if src.allow_inf_in_output:
        self.allow_inf_in_output = True

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(ReinterpretAndCropLayer, self).get_dep_layers()
    if self.size_base:
      deps.append(self.size_base)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param get_layer:
    """
    super(ReinterpretAndCropLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("size_base"):
      d["size_base"] = get_layer(d["size_base"])

  @classmethod
  def get_out_data_from_opts(cls, name, sources,
                             switch_axes=None, size_base=None, set_axes=None,
                             enforce_batch_major=False, enforce_time_major=False,
                             set_sparse=None, set_sparse_dim=NotSpecified, increase_sparse_dim=None,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|list[str] switch_axes: e.g. "bt" to switch batch and time axes
    :param LayerBase|None size_base: similar as size_target
    :param dict[str,int] set_axes:
    :param bool enforce_batch_major:
    :param bool enforce_time_major:
    :param bool|None set_sparse: if bool, set sparse value to this
    :param int|None|NotSpecified set_sparse_dim: set sparse dim to this. assumes that it is sparse
    :param int|None increase_sparse_dim: add this to the dim. assumes that it is sparse
    """
    out = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert not (enforce_batch_major and enforce_time_major)
    if enforce_batch_major:
      out = out.copy_as_batch_major()
    if enforce_time_major:
      out = out.copy_as_time_major()

    def map_axis_name(s):
      """
      :param str s:
      :rtype: str
      """
      if s.upper() == "B":
        return "batch_dim_axis"
      if s.upper() == "T":
        return "time_dim_axis"
      if s.upper() == "F":
        return "feature_dim_axis"
      assert s in ["batch_dim_axis", "time_dim_axis", "feature_dim_axis"]
      return s

    if switch_axes:
      assert len(switch_axes) == 2
      axes_s = list(map(map_axis_name, switch_axes))
      axes = [getattr(out, s) for s in axes_s]
      for i in range(len(axes)):
        setattr(out, axes_s[i], axes[(i + 1) % len(axes)])
    if set_axes:
      for s, i in sorted(set_axes.items()):
        s = map_axis_name(s)
        if isinstance(i, int):
          assert enforce_batch_major or enforce_time_major, "%r: explicit set_axes %r" % (name, set_axes)
        i = out.get_axis_from_description(i)
        setattr(out, s, i)
        if s == "feature_dim_axis":
          out.dim = out.batch_shape[out.feature_dim_axis]
    if size_base:
      out.size_placeholder = size_base.output.size_placeholder.copy()
    if set_sparse is not None:
      assert isinstance(set_sparse, bool)
      out.sparse = set_sparse
    if set_sparse_dim is not NotSpecified:
      assert set_sparse_dim is None or isinstance(set_sparse_dim, int)
      out.dim = set_sparse_dim
    if increase_sparse_dim:
      assert out.sparse
      out.dim += increase_sparse_dim
    return out

################### ADDED ######################
### Variant of class::WindowLayer() in TFNetworkLayer.py
class StridedWindowLayer(_ConcatInputLayer):
  """
  Adds a window dimension.
  By default, uses the time axis and goes over it with a sliding window.
  The new axis for the window is created right after the time axis.
  Will always return as batch major mode.
  E.g. if the input is (batch, time, dim), the output is (batch, time, window_size, dim).
  If you want to merge the (window_size, dim) together to (window_size * dim,),
  you can use the MergeDimsLayer, e.g. {"class": "merge_dims", "axes": "except_time"}.

  This is not to take out a window from the time-dimension.
  See :class:`SliceLayer` or :class:`SliceNdLayer`.
  """
  layer_class = "strided_window"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  def __init__(self, window_size, window_left=None, window_right=None, stride=1, axis="T", padding="same", **kwargs):
    """
    :param int window_size:
    :param int|None window_left:
    :param int|None window_right:
    :param str|int axis: see Data.get_axis_from_description()
    :param str padding: "same" or "valid"
    :param kwargs:
    """
    super(StridedWindowLayer, self).__init__(**kwargs)
    data = self.input_data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      assert self._rec_previous_layer, "%s: expected to be used inside a RecLayer" % self
      assert padding == "same"
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]  # (batch,window,...)
      next_state = tf.concat(
        [prev_state[:, 1:], tf.expand_dims(data.placeholder, axis=1)], axis=1)  # (batch,window,...)
      self.rec_vars_outputs["state"] = next_state
      self.output.placeholder = next_state
    else:
      axis = data.get_axis_from_description(axis)
      from returnn.tf.util.basic import windowed_nd
      #self.output.placeholder = windowed_nd(
      #  data.placeholder,
      #  window_size=window_size, window_left=window_left, window_right=window_right,
      #  padding=padding, time_axis=axis, new_window_axis=axis + 1)
      self.output.placeholder = tf.signal.frame(
        data.placeholder,
        frame_length=window_size, frame_step=stride,
        pad_end=(padding=="same"), axis=axis
        ) 
    self.output.placeholder.set_shape(tf.TensorShape(self.output.batch_shape))
    self.output.size_placeholder = self.input_data.size_placeholder.copy()
    axis_wo_b = self.output.get_batch_axis_excluding_batch(axis)
    if axis_wo_b in self.output.size_placeholder:
      self.output.size_placeholder[axis_wo_b] = ConvLayer.calc_out_dim(
        in_dim=self.output.size_placeholder[axis_wo_b],
        filter_size=window_size, stride=stride, dilation_rate=1, padding=padding)
        #filter_size=window_size, stride=1, dilation_rate=1, padding=padding)

  @classmethod
  def get_out_data_from_opts(cls, name, window_size, axis="T", sources=(), **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int window_size:
    :param str axis:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources)
    data = data.copy_template(name="%s_output" % name)
    data = data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      axis = 0
    else:
      axis = data.get_axis_from_description(axis)
    data = data.copy_add_spatial_dim(spatial_dim_axis=axis + 1, dim=window_size)  # add new axis right after
    return data

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, window_size, axis="T", sources=(), **kwargs):
    """
    :param tf.Tensor batch_dim:
    :param TFNetworkRecLayer.RecLayer|LayerBase rec_layer:
    :param int window_size:
    :param str axis:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    data = get_concat_sources_data_template(sources)
    data = data.copy_as_batch_major()
    if axis == "T" and data.time_dim_axis is None:
      # Assume inside RecLayer.
      shape = list(data.batch_shape)
      shape[0] = batch_dim
      shape.insert(1, window_size)
      return {"state": tf.zeros(shape, dtype=data.dtype)}
    return {}


