#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:12:42 2019

@author: renato
"""


# #########################################################################################################################
#Loading document corpus
docMap =  {}
#
fil = open('/home/joao/datasets/conll/corpus.out.tsv','r',1,encoding='utf-8')

for line in fil:
      elems = line.split("\t")
      titulo = elems[0]
      if len(elems) == 1 :
            continue
      conteudo = elems[1]
      conteudo = conteudo.replace("\\s+", " ")
      conteudo = conteudo.strip()
      titulo = titulo.lower()
      docMap[titulo] = conteudo
print("done loading data set.")