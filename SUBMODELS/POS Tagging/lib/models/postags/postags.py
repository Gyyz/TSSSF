#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser
from lib.models.postags.basetags import Basetags

#***************************************************************
class Postags(Basetags):
  """"""
  
  #=============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    
    vocabs = dataset.vocabs
    inputs = dataset.inputs
    targets = dataset.targets
    
    reuse = (moving_params is not None)
    self.tokens_to_keep3D = tf.expand_dims(tf.to_float(tf.greater(inputs[:,:,0], vocabs[0].ROOT)), 2)
    self.sequence_lengths = tf.reshape(tf.reduce_sum(self.tokens_to_keep3D, [1, 2]), [-1,1])
    self.n_tokens = tf.reduce_sum(self.sequence_lengths)
    self.moving_params = moving_params
    
    word_inputs = vocabs[0].embedding_lookup(inputs[:,:,0], inputs[:,:,1], moving_params=self.moving_params)
#    tag_inputs  = vocabs[1].embedding_lookup(inputs[:,:,2], moving_params=self.moving_params)

    top_recur = self.embed_concat(word_inputs)#, tag_inputs)
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)

    with tf.variable_scope('Tags', reuse = reuse):
      input_size = top_recur.get_shape().as_list()[-1]
      batch_size = tf.shape(top_recur)[0]
      bucket_size = tf.shape(top_recur)[1]
      input_shape = tf.pack([batch_size, bucket_size, input_size])
      if reuse is None:
        top_recur = tf.nn.dropout(top_recur, 0.6, seed=666)
#      top_recur = tf.reshape(top_recur, input_shape)
      weight_pos = tf.get_variable('pos', [input_size, len(vocabs[1])], initializer=tf.random_normal_initializer())
      bias_pos = tf.get_variable('pos_bias', [1], initializer=tf.zeros_initializer)
      top_recur = tf.reshape(top_recur, [-1, input_size])
      pos_logits = tf.matmul(top_recur, weight_pos) + bias_pos
      pos_logits = tf.reshape(pos_logits, tf.pack([batch_size, bucket_size,len(vocabs[1])]))
      output = self.posout(pos_logits, targets[:,:,0])
      output['embed'] = tf.pack([word_inputs])
      output['recur'] = top_recur
      output['pos_logits'] = pos_logits

    return output

  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
