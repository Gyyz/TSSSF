#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter

from lib.etc.k_means import KMeans
from configurable import Configurable
from vocab import Vocab
from metabucket import Metabucket

#***************************************************************
class Dataset(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, filename, vocabs, builder, *args, **kwargs):
    """"""
    
    super(Dataset, self).__init__(*args, **kwargs)
    self._file_iterator = self.file_iterator(filename)
    self._train = (filename == self.train_file)
    self._metabucket = Metabucket(self._config, n_bkts=self.n_bkts)
    self._data = None
    self.vocabs = vocabs
    self.rebucket()
    
    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None,None,None), name='inputs')
    self.targets = tf.placeholder(dtype=tf.int32, shape=(None,None), name='targets')
    self.sntmod = tf.placeholder(dtype=tf.float32,shape=(None, 3),name='sntmod')

    self.builder = builder()
  
  #=============================================================
  def file_iterator(self, filename):
    """"""
    
    with open(filename) as f:
      if self.lines_per_buffer > 0:
        buff = [[]]
        while True:
          line = f.readline()
          while line:
            line = line.strip().split()
            if line:
              buff[-1].append(line)
            else:
              if len(buff) < self.lines_per_buffer:
                if buff[-1]:
                  buff.append([])
              else:
                break
            line = f.readline()
          if not line:
            f.seek(0)
          else:
            buff = self._process_buff(buff)
            yield buff
            line = line.strip().split()
            if line:
              buff = [[line]]
            else:
              buff = [[]]
      else:
        buff = [[]]
        for line in f:
          line = line.strip().split()
          if line:
            buff[-1].append(line)
          else:
            if buff[-1]:
              buff.append([])
        if buff[-1] == []:
          buff.pop()
        buff = self._process_buff(buff)
        while True:
          yield buff
  
  #=============================================================
  def _process_buff(self, buff):
    """"""
    
    words, tags = self.vocabs
    for i, sent in enumerate(buff):
      targetflag=0
      for j, token in enumerate(sent):
        if token[2] != 'o':
          targetflag=1
        word, tag, istarget, bftarget, aftarget, sentmod = token[0], token[1], 0 if token[2] == 'o' else 1, 1 if token[2] =='o' and targetflag==0 else 0, 1 if token[2] == 'o' and targetflag == 1 else 0, self.getmood(token[2])
        buff[i][j] = (word,) + words[word] + tags[tag] + (int(istarget),) + (int(bftarget),) + (int(aftarget),) + (sentmod,)
    return buff
  
  #=============================================================
  def getmood(self, polority):
    """"""
    if polority == 'o':
      return 0
    else:
      polority = polority.split('-')[1]
      if polority == 'positive':
        return 2
      elif polority == 'negative':
        return 4
      else:
        return 6
  #=============================================================
  def reset(self, sizes):
    """"""
    
    self._data = []
    self._targets = []
    self._metabucket.reset(sizes)
    return
  
  #=============================================================
  def rebucket(self):
    """"""
    
    buff = self._file_iterator.next()
    len_cntr = Counter()
    
    for sent in buff:
      len_cntr[len(sent)] += 1
    self.reset(KMeans(self.n_bkts, len_cntr).splits)
    
    for sent in buff:
      self._metabucket.add(sent)
    self._finalize()
    return
  
  #=============================================================
  def _finalize(self):
    """"""
    
    self._metabucket._finalize()
    return
  
  #=============================================================
  def get_minibatches(self, batch_size, input_idxs, target_idxs, shuffle=True):
    """"""
    
    minibatches = []
    for bkt_idx, bucket in enumerate(self._metabucket):
      if batch_size == 0:
        n_splits = 1
      #elif not self.minimize_pads:
      #  n_splits = max(len(bucket) // batch_size, 1)
      #  if bucket.size > 100:
      #    n_splits *= 2
      else:
        n_tokens = len(bucket) * bucket.size
        n_splits = max(n_tokens // batch_size, 1)
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      arr_sp = np.array_split(range_func(len(bucket)), n_splits)
      for bkt_mb in arr_sp:
        if len(bkt_mb)>0:
          minibatches.append( (bkt_idx, bkt_mb) )
    if shuffle:
      np.random.shuffle(minibatches)
    for bkt_idx, bkt_mb in minibatches:
      data = self[bkt_idx].data[bkt_mb]
      sents = self[bkt_idx].sents[bkt_mb]
      sntmodp = self[bkt_idx].smod[bkt_mb]
      maxlen = np.max(np.sum(np.greater(data[:,:,0], 0), axis=1))
     # print("[tlog] maxlen\n"+str(maxlen))

      feed_dict = {
        self.inputs: data[:,:maxlen,input_idxs],
        self.targets: data[:,:maxlen,target_idxs],

        self.sntmod: sntmodp
      }
      yield feed_dict, sents

  #=============================================================
  @property
  def n_bkts(self):
    if self._train:
      return super(Dataset, self).n_bkts
    else:
      return super(Dataset, self).n_valid_bkts
  
  #=============================================================
  def __getitem__(self, key):
    return self._metabucket[key]
  def __len__(self):
    return len(self._metabucket)

