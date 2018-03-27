# Targeted Sentiment Analysis

This repository is the code for paper [*Implicit Syntactic Features for Targeted Sentiment Analysis*](http://www.aclweb.org/anthology/I17-1052), it contains the modified code used to train the targeted sentiment analysis. We mainly employ the tensorflow LSTM framework in the paper [*deep biaffine attention for neural dependency parsing*](https://arxiv.org/abs/1611.01734). [[CODE](https://github.com/tdozat/Parser)] All our modified codes mainly inlude `/lib/models/` dirs, `network.py`, `dataset.py`, `bucket.py` and `vocab.py`  parts of the original CODE

## 1. Pre-trainning Submodels

----------------

### We pretrain the submodels on PTB data:

#### 1.1 For POS Tagging model:

See the `Readme.md` in `./SUBMODELS/POS/` dirs

#### 1.2 For Dependency Parser model:

See the `Readme.md` in `./SUBMODELS/Dependency Parsing/` dirs



## 2. Sentiment Analysis

------------------------

### Syntactic models:

#### 2.1 Save pre-trained models:

We saves the best result models for both POS tagging model and Dependency Parsing model( include normal dependency model and non-postags dependency model) as the pre-trained models.

#### 2.2  Re-Load the pre-trained models and train sentiment analysis model: 

See the `Readme.md` in `./Sentiment/` dirs

