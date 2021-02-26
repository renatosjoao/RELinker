#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:13:06 2019

@author: renato
"""


# Loading Wikipedia corpus
WikiMap = {}
#
bff  = open("/home/joao/Wiki2016/wiki.en.tsv",'r',1,encoding='utf-8')
for line in bff:
      elems = line.split("\t")
      titulo = elems[0]
      if len(elems) == 1 :
            continue
      conteudo = elems[1];
      conteudo = conteudo.replace("\\s+", " ")
      conteudo = conteudo.strip()
      titulo = titulo.lower()
      WikiMap[titulo] = conteudo
print("done loading Wikipedia.")