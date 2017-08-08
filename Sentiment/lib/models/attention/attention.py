#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser
from lib.models.attention.baseatt import BaseAttentions
#***************************************************************
class Attention(BaseAttentions):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets 
    targets_vec = dataset.sntmod    
    reuse = (moving_params is not None)
    # self.reuse = reuse
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    
    if self.load_emb:
      if self.stack:
        word_inputs_top = vocabs[0].embedding_lookup(inputs[:,:,1], pret_inputs_stack=inputs[:,:,3], moving_params=self.moving_params, top=True)
        word_inputs_btm = vocabs[0].embedding_lookup(inputs[:,:,0], pret_inputs=inputs[:,:,2], moving_params=self.moving_params)
        tag_inputs_top  = vocabs[1].embedding_lookup(inputs[:,:,4], moving_params=self.moving_params, top=True)
        tag_inputs_btm  = vocabs[1].embedding_lookup(inputs[:,:,4], moving_params=self.moving_params)
        istarget = inputs[:,:,5]
        bftarget = inputs[:,:,6]
        aftarget = inputs[:,:,7]
      else:
        word_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
        tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)
    else:
      word_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], moving_params=self.moving_params)
      tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,1], moving_params=self.moving_params)
    
    if self.stack:
      top_recur = self.embed_concat(word_inputs_btm, tag_inputs_btm)
    else:
      top_recur = self.embed_concat(word_inputs, tag_inputs)

    # Bottom/Original RNN layers
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)

    top_mlp = top_recur

    if self.n_mlp > 0:
      with tf.variable_scope('MLP0', reuse=reuse):
        dep_mlp, head_dep_mlp, rel_mlp, head_rel_mlp = self.MLP(top_mlp, n_splits=4)
      for i in xrange(1,self.n_mlp):
        with tf.variable_scope('DepMLP%d' % i, reuse=reuse):
          dep_mlp = self.MLP(dep_mlp)
        with tf.variable_scope('HeadDepMLP%d' % i, reuse=reuse):
          head_dep_mlp = self.MLP(head_dep_mlp)
        with tf.variable_scope('RelMLP%d' % i, reuse=reuse):
          rel_mlp = self.MLP(rel_mlp)
        with tf.variable_scope('HeadRelMLP%d' % i, reuse=reuse):
          head_rel_mlp = self.MLP(head_rel_mlp)
    else:
      dep_mlp = head_dep_mlp = rel_mlp = head_rel_mlp = top_mlp

#==========================================================================================================================
    with tf.variable_scope('Attention_based',reuse=reuse):
      top_att_recur= tf.concat(2,[dep_mlp, head_dep_mlp, rel_mlp, head_rel_mlp, word_inputs_top])#top_mlp
#      print(top_att_recur.get_shape().as_list())
      with tf.variable_scope('AttRnn',reuse=reuse):
        top_att_recur, _ = self.RNN(top_att_recur)
#      print(top_att_recur.get_shape().as_list())
      htscore = self.getTarHd(top_att_recur, istarget)
      alphaSeq = self.cptSnt(htscore, top_att_recur, attscope='atb')
      sntVec = self.Seq2Pb(alphaSeq)
      attout = self.attoutput(sntVec, targets_vec)
      return attout
#======================================================================================================================    
# 
#  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
