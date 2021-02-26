#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:12:43 2019

@author: renato
"""


(#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Renato Stoffalette Joao ( renatosjoao@gmail.com )
# Copyright 2019 @ Renato

import gensim
import os
import re
from gensim.models import Doc2Vec
from gensim.models.doc2vec import  TaggedDocument
import multiprocessing
from collections import OrderedDict
import logging
import sys
from random import shuffle
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub(' ', string)

def _remove_stop_words(string,sw=[]):
    return ' '.join([word if word not in sw else '' \
                     for word in string.strip().split(' ')])

def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case
    return re.sub('\s+',' ',string).strip().lower()



if __name__ == '__main__':
      program = os.path.basename(sys.argv[0])
      logger = logging.getLogger(program)

      logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
      logging.root.setLevel(level=logging.INFO)
      logger.info("running %s" % ' '.join(sys.argv))


      #########################################################################################################################
      #Loading document corpus
      docMap =  {}

      fil = open('/home/joao/datasets/conll_corpus/corpus.out.tsv','r',1,encoding='utf-8')

      for line in fil:
          elems = line.split("\t")
          titulo = elems[0]
          if len(elems) == 1 :
             continue
          conteudo = elems[1]
          conteudo = conteudo.replace("\\s+", " ")
          conteudo = conteudo.lower().strip()
          titulo = titulo.lower().strip()
          docMap[titulo] = conteudo
      print("done loading data set.")


      begin = time()
      alldocs = []

      topTitles = set()
      wiki_top = open('/home/joao/Wiki2016/mentions.50.txt', 'r',1,encoding='utf-8')

      for linha in wiki_top:
          linha = linha.strip()
          topTitles.add(linha)
      print("Num of titles "+str(len(topTitles)))


      wiki_f = open('/home/joao/Wiki2016/wiki.en.tsv', 'r',1,encoding='utf-8')

      num_titles = 0
      print('loading  wikipedia docs...')
      for line in wiki_f:
          splitline = line.split('\t')

          title = splitline[0]
          conteudo = splitline[1]
          if title in topTitles:
              num_titles += 1
              sentences = conteudo.split()

              if len(sentences) >= 200:
                  sentences = sentences[:200]
                  alldocs.append(TaggedDocument(sentences, [title]))
              else:
                  alldocs.append(TaggedDocument(sentences, [title]))


      for t,c in docMap.items():
            sentences = c.split()
            alldocs.append(TaggedDocument(sentences, [t]))


      cores = multiprocessing.cpu_count()
      assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

      cores = multiprocessing.cpu_count()
# =============================================================================
#     model = Doc2Vec(size=300, window=10, min_count=10, alpha=0.05, min_alpha=0.025, workers=16,dm=0, negative=5, sample=0 )
# =============================================================================

      #PV-DBOW plain
      #PV_DBOW =  Doc2Vec(dm=0, size=300, window=15, negative=5, hs=0, min_count=10, sample=0, workers=18)
      #PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
      #PV_DM = Doc2Vec(dm=1, size=300, window=5, negative=5, hs=0, min_count=5, sample=0, workers=20 ,alpha=0.01)
      #PV-DM w/ concatenation - big, slow, experimental mode
      #window=5 (both sides) approximates paper's apparent 10-word total window size
      PV_DM_CONCAT = Doc2Vec(dm=1, dm_concat=1, size=300, window=5, negative=5, hs=0, min_count=10, sample=0, workers=18)



      #PV_DBOW.build_vocab(alldocs)
      #print("%s vocabulary scanned & state initialized PV_DBOW")
      #PV_DM.build_vocab(alldocs)
      #print("%s vocabulary scanned & state initialized PV_DM")
      PV_DM_CONCAT.build_vocab(alldocs)
      print("%s vocabulary scanned & state initialized PV_DM_CONCAT")

      #print("Training PV_DBOW")
      #PV_DBOW.train(alldocs, total_examples=PV_DBOW.corpus_count, epochs=500)
      #print("Training PV_DM")
      #PV_DM.train(alldocs, total_examples=PV_DM.corpus_count,epochs=500)

      print("Training PV_DM_CONCAT")
      PV_DM_CONCAT.train(alldocs, total_examples=PV_DM_CONCAT.corpus_count, epochs=500)


#      models_by_name = OrderedDict((str(PV_DBOW), PV_DBOW) for model in simple_models)

      #model_dbow_dmm = ConcatenatedDoc2Vec([PV_DBOW,PV_DM])
      #model_dbow_dmc = ConcatenatedDoc2Vec([PV_DBOW, PV_DM_CONCAT])





      #PV_DBOW.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.PV_DBOW.min10words.20words.top50")
      #PV_DM.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.PV_DM.min5words.200words.top50")
      PV_DM_CONCAT.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.PV_DM_CONCAT.min10words.200words.top50")


      #model_dbow_dmm.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.model_dbow_dmm.min10words")
      #model_dbow_dmc.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.model_dbow_dmc.min10words")

      #model.save("/home/joao/Wiki2016/model/doc2vec_gensim.min10words.dm0")

	# load the doc2vec
	#model = Doc2Vec.load("/home/joao/Wiki2016/model/doc2vec_gensim")
	#print(model.docvecs.most_similar(["human interface"], topn=2))
	#print(model.docvecs['alan turing'])
    #print()

	#get the raw embedding for that sentence as a NumPy vector:
	#print model["SENT_0"]

    # n_similarity computes the cosine similarity in Doc2Vec
    #score = model.n_similarity(questions1_split[i],questions2_split[i])