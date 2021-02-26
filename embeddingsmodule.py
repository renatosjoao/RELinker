#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:07:35 2019

@author: renato
"""


## demo how to load the word2vec model from WIKIPEDIA2VEC

import numpy as np
from numpy import array
from gensim.models.keyedvectors import KeyedVectors
from timeit import default_timer as timer
import gensim
import math
import sys,os
import logging
from math import*
from decimal import Decimal
from wikipedia2vec import Wikipedia2Vec

WIKIPEDIA_URI_BASE = u"https://en.wikipedia.org/wiki/{}"



def loadEmbedding():
    #vecmodel1987 = KeyedVectors.load_word2vec_format('./enwiki_20180420_100d.txt', binary=False)
    embedding = Wikipedia2Vec.load('./embeddings/enwiki_20180420_500d.pkl')
    #embedding = Wikipedia2Vec.load('./embeddings/enwiki_20180420_100d.pkl')
    return embedding
    

#embedding = loadEmbedding()
#amaz = embedding.get_word_vector('amazon')


# manually compute cosine similarity
#dot = np.dot(amaz,amaz)
#norma = np.linalg.norm(amaz)
#normb = np.linalg.norm(amaz)
#cos = dot / (norma * normb)

#print(cos)



#embedding.get_entity_vector('Johansson')
