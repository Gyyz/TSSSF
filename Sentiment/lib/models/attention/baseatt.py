#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
from vocab import Vocab
from lib.models import NN


# ***************************************************************
class BaseAttentions(NN):
  """"""

  # =============================================================
  def __call__(self, dataset, moving_params=None):
    """"""
    raise NotImplementedError

  # =============================================================
  def prob_argmax(self, parse_probs, rel_probs, tokens_to_keep):
    """"""
    raise NotImplementedError

  #==============================================================
  def getTarHd(self, word_hlstm, targetwd_idxs, comput_idxs=None):
    """
    :param word_hlstm: the hidden lstm output
    :param comput_idxs: the id information of the word t
    :param targetwd_idxs: the id information of target wordo compute
    :return: the attention lstm concat with target attention( whole snt/ before target words snt/ after target words snt)
    """
    # Get shape
    batch_size = tf.shape(word_hlstm)[0]
    bucket_size = tf.shape(word_hlstm)[1]
    lstm_dim = word_hlstm.get_shape().as_list()[-1]
    input_shape = tf.pack([batch_size, bucket_size, lstm_dim])
    target_wd_shape = tf.pack([batch_size, bucket_size, 1])
    lstm_shapes = tf.ones(input_shape)

    # Get attention matrix
    target_wd_matrix = tf.mul(lstm_shapes, tf.to_float(tf.reshape(targetwd_idxs, target_wd_shape)))
    raw_score = tf.mul(word_hlstm, tf.reshape(tf.to_float(targetwd_idxs), target_wd_shape))
    word_hlstm_sum = tf.reduce_sum(raw_score, axis=1)
    ht_num = tf.to_float(tf.reduce_sum(targetwd_idxs, axis=1))
    ht_score = tf.transpose(word_hlstm_sum) / ht_num
    ht_score = tf.transpose(ht_score)

    # Transfer score
    nontar_matrix = 1 - target_wd_matrix
    nontarget_tr = tf.transpose(nontar_matrix, (1, 0, 2))
    ht_3D_score = tf.mul(nontarget_tr, ht_score)
    ht_3Dscore = tf.transpose(ht_3D_score, (1, 0, 2))

    # Process the ht_score
    nontarget_lstm = tf.mul(word_hlstm, nontar_matrix)
    wd_att = tf.concat(2, [nontarget_lstm, ht_3Dscore])
    # wd_att=tf.concat(2,[word_hlstm, ht_3Dscore])

    #choose comput_idxs
    if comput_idxs is not None:
      wd_att = tf.mul(wd_att, tf.to_float(tf.reshape(comput_idxs, target_wd_shape)))

    return wd_att

  #compute the belta and alpha, get the sent vector representation
  def cptSnt(self, att_lstm, wd_lstm, attscope=None):
    """
    :param att_lstm: the word lstm concat with target attention
    :param wd_lstm: the hidden output of lstm
    :param scope: the variable scope(weights)
    :return: the sentence representation with a alpha sequence
    """
    # batch_size bucket_size input_size
    batch_size = tf.shape(att_lstm)[0]
    bucket_size = tf.shape(att_lstm)[1]
    input_size = att_lstm.get_shape().as_list()[-1]
    output_shape = tf.pack([batch_size, bucket_size, 1])
    att_lstm = tf.reshape(att_lstm, tf.pack([batch_size, bucket_size, input_size]))

    # Get the weight matrix and do multiply
    with tf.variable_scope(attscope):
      weight_almat = tf.get_variable('alpha_matrix', [input_size, 1], initializer=tf.random_normal_initializer())
      bias_almat = tf.get_variable('att_bias', [1], initializer=tf.zeros_initializer)
      att_lstm_rp = tf.reshape(att_lstm, [-1, input_size])
      raw_belta = tf.matmul(att_lstm_rp, weight_almat) + bias_almat
      wd_belta = tf.tanh(raw_belta)

      # Get alpha and the sentence representation
      wd_alpha = tf.reshape(wd_belta, output_shape)
      wd_salpha = tf.nn.softmax(wd_alpha, 1)
      AlphaSeq = tf.mul(wd_lstm, wd_salpha)
      AlphaRpt = tf.reduce_sum(AlphaSeq, 1)

    return AlphaRpt

  # compute the alpha sentence representation
  def Seq2Pb(self, alpha_seqt, alpha_seql=None, alpha_seqr=None, atscope=None, output_size=3, gates=False):
    """
    :param alpha_seqt: the whole sentence alpha representation
    :param alpha_seql: the alpha representation before the target words
    :param alpha_seqr: the alpha representation after the target words
    :param gates: use gates to adjust the three alpha sequence
    :return: the probablity vector of sentiment
    """
    SentiVector=None
    if (alpha_seql is not None) and (alpha_seqr is not None) and not gates:
      with tf.variable_scope(atscope):
        if tf.shape(alpha_seqt) != tf.shape(alpha_seqr):
          raise ValueError('Indices should have the same shape [batch_size, 2*recur_size]')
        #Get the input shape
        input_shape = tf.shape(alpha_seqt)
        batch_size=input_shape[0]
        input_size = alpha_seqt.get_shape().as_list()[-1]
        output_shape=[]
        output_shape.append(batch_size)
        output_shape.append(output_size)
        output_shape=tf.pack(output_shape)

        #Get the matrix
        weight_matt = tf.get_variable('htattt',[input_size, output_size], initializer=tf.random_normal_initializer())
        weight_matl = tf.get_variable('htattl', [input_size, output_size], initializer=tf.random_normal_initializer())
        weight_matr = tf.get_variable('htattr', [input_size, output_size], initializer=tf.random_normal_initializer())
        weight_bias = tf.get_variable('attbias', [output_size], initializer=tf.random_normal_initializer())

        SentiVector= tf.matmul(alpha_seqt, weight_matt) + tf.matmul(alpha_seql, weight_matl) + tf.matmul(alpha_seqr, weight_matr) + weight_bias

    elif (alpha_seql is not None) and (alpha_seqr is not None) and gates:
      with tf.variable_scope(atscope):
        # Get the input shape
        input_shape = tf.shape(alpha_seqt)
        batch_size = input_shape[0]
        input_size = alpha_seqt.get_shape().as_list()[-1]
        output_shape = []
        output_shape.append(batch_size)
        output_shape.append(output_size)
        output_shape = tf.pack(output_shape)

    else:
      with tf.variable_scope('sntvec' or atscope):
        # Get the input shape
        ndims = len(alpha_seqt.get_shape().as_list())
        input_shape = tf.shape(alpha_seqt)
        input_size = alpha_seqt.get_shape().as_list()[-1]
        output_shape = []
        batch_size = input_shape[0]
        output_shape.append(input_shape[0])
        output_shape.append(output_size)
        output_shape = tf.pack(output_shape)
        alpha_seqt = tf.reshape(alpha_seqt, tf.pack([batch_size, input_size]))

        # Get the Matrix and Multiply
        weight_matrix = tf.get_variable('AttSent', [input_size, output_size],
                                            initializer=tf.random_normal_initializer())
        bias_matrix = tf.get_variable('AttSentBias', [3], initializer=tf.zeros_initializer)
        SentiVector = tf.matmul(alpha_seqt, weight_matrix) + bias_matrix

          # softmax the result
            #  SentiPb = tf.nn.softmax(SentiVector)
    return SentiVector

  #=========================================================================================
  def attoutput(self, predlogits2D, glodenvec2D):
    """
    :param predlogits2D: the model output logits
    :param glodenvec2D:  target sentiment vec
    :return: the output dict
    """
    original_shape = tf.shape(predlogits2D)
    sent_num = original_shape[0]
    probabilities_2D = tf.nn.softmax(predlogits2D)

    logits1D = tf.to_int32(tf.argmax(predlogits2D, 1))
    targets1D = tf.to_int32(tf.argmax(glodenvec2D, 1))
    correct1D = tf.to_int32(tf.equal(logits1D, targets1D))
    n_correct = tf.reduce_sum(correct1D)
    accuracy = n_correct / sent_num

    cross_entropy1D = tf.nn.softmax_cross_entropy_with_logits(predlogits2D, glodenvec2D)
    loss = tf.reduce_mean(cross_entropy1D)

    
#    predictCounter = Counter(np.argmax(predlogits2D))
#    goldenCounter = Counter(np.argmax(glodenvec2D))

#    posCP = posCT = 0
#    negCP = negCT = 0
#    neuCP = neuCT = 0

#    posCP, negCP, neuCP = predictCounter.values()
#    posCT, negCT, neuCT = goldenCounter.values()

#    posCorrect = negCorrect = neuCorrect = 0

#    for pred, targ in zip(logits1D, targets1D):
#      if pred == 0 and targ == 0:
#        posCorrect += 1
#      if pred == 1 and targ == 1:
#        negCorrect += 1
#      if pred == 2 and targ == 2:
#        neuCorrect += 1
    
#    posF1 = self.getF1(posCorrect, posCP, posCT)
#    negF1 = self.getF1(negCorrect, negCP, negCT)
#    neuF1 = self.getF1(neuCorrect, neuCP, neuCT)
    
    attoutput = {
        'probabilities': probabilities_2D,
        'predictions': logits1D,
        'batch_size': sent_num,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'loss': loss}#,
#        'Fvalues': (posF1, negF1, neuF1)}
    return attoutput
  #==============================================================
  def getF1(self, correct, predict, golden):
    """
    :param correct: 
    :param predict: 
    :param golden: 
    :return: 
    """
    if correct == 0:
      return 0
    P = correct / predict
    R = correct / golden
    return P * R * 2 / (P + R)

  #=========================================================================================
  def validate(self, dt_inputs, dt_targets, dt_probs):
    """
    :param dt_input: the model input
    :param dt_targets:  target Gold
    :param dt_probs: targetPredict
    :return: the output dict
    """
    sents = []
    dt_targets = np.argmax(dt_targets, 1)
    dt_probs = np.argmax(dt_probs, 1)
    for inputs, targets, probs in zip(dt_inputs, dt_targets, dt_probs):
      tokens_to_keep = np.greater(inputs[:,0], Vocab.ROOT)
      length = np.sum(tokens_to_keep)
      sent = -np.ones( (length, 10), dtype=int)
      tokens = np.arange(0, length)
      sent[:,0:8] = inputs[tokens]
      sent[:,8] = targets
      sent[:,9] = probs
    return sents

  #========================================================================================
   


  @property
  def input_idxs(self):
    if self.load_emb:
      if self.stack: # self.extra_emb:
        return (0, 1, 2, 3, 4, 5, 6, 7)
      else:
        return (0, 1, 2)
    else:
      return (0, 1)
  @property
  def target_idxs(self):
    if self.load_emb:
      if self.stack: # self.extra_emb:
        return (8)
      else:
        return (3, 4, 5)
    else:
      return (2, 3, 4)

