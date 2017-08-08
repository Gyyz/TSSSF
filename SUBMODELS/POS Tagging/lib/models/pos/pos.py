#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from vocab import Vocab
from lib.models.parsers.base_parser import BaseParser

#***************************************************************
class POS(BaseParser):
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
    tags  = inputs[:,:,2]
    
    top_recur = self.embed_concat(word_inputs)
    for i in xrange(self.n_recur):
      with tf.variable_scope('RNN%d' % i, reuse=reuse):
        top_recur, _ = self.RNN(top_recur)
    
    with tf.variable_scope('POS', reuse=reuse):
      top_recur = self.MLP(top_recur)
      input_size = top_recur.get_shape().as_list()[-1]
      batch_size = tf.shape(top_recur)[0]
      bucket_size = tf.shape(top_recur)[1]
      input_shape = tf.pack([batch_size, bucket_size, input_size])

      hdstate = tf.reshape(top_recur, input_shape)

      clsWeight = tf.get_variable('softmax', [input_size, len(vocab[1])], initializer=tf.random_normal_initializer())
      clsBias = tf.get_variable('softmaxbias', [len(vocab[1])], initializer=tf.zero_initializer)

      POS_logits = tf.matmul(hdstate, clsWeight)+clsBias

      POS_label = tf.softmax(POS_logits, axis=1)

      POS_out = self.output(POS_logits, tags)

      output = {}

      output['predictions'] = POS_label
      output['correct'] = POS_out['correct']
      output['tokens'] = POS_out['tokens']
      output['n_correct'] = tf.reduce_sum(output['correct'])
      output['n_tokens'] = self.n_tokens
      output['accuracy'] = POS_out['n_correct'] / POS_out['n_tokens']
      output['loss'] = POS_out['loss']
      output['POS_logits'] = POS_logits
   
    return output
  
  #=============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    
    parse_preds = self.parse_argmax(parse_probs, tokens_to_keep)
    rel_probs = rel_probs[np.arange(len(parse_preds)), parse_preds]
    rel_preds = self.rel_argmax(rel_probs, tokens_to_keep)
    return parse_preds, rel_preds
