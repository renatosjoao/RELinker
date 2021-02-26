#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:13:28 2019

@author: renato
"""
# -*- coding: utf-8 -*-
# Author: Renato Stoffalette Joao ( renatosjoao@gmail.com )
# Copyright 2019 @ Renato

import gensim
from gensim.models import Word2Vec
import logging
import multiprocessing
import os
import re
import sys

from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub(' ', string)


def _remove_stop_words(string,sw=[]):
    return ' '.join([word if word not in sw else '' \
                     for word in string.strip().split(' ')])

def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case
    return re.sub('\s+',' ',string).strip().lower()

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    wiki_f = open('/home/joao/Wiki2016/wiki.en.tsv', 'r',1,encoding='utf-8')

    topTitles = set()
    wiki_top = open('/home/joao/Wiki2016/mentions.50.txt', 'r',1,encoding='utf-8')

    for linha in wiki_top:
        linha = linha.strip()
        topTitles.add(linha)
    print("Num of titles "+str(len(topTitles)))

    alldocs = []
    num_titles = 0
    print('loading  wikipedia docs...')
    for line in wiki_f:
        splitline = line.split('\t')
        title = splitline[0]
        conteudo = splitline[1]
        num_titles += 1
        sentences = conteudo.split()

        if len(sentences) > 20:
            sentences = sentences[:20]
            alldocs.append(sentences)
    print('...done.')
    print(num_titles)
    print(str(len(alldocs)))

    begin = time()
    cores = multiprocessing.cpu_count()

    model = Word2Vec(alldocs,size=300,window=10,min_count=10, alpha=0.025, min_alpha=0.0001, iter=500,  workers=12, sg=1)
	#model.build_vocab(alldocs)
	#model.train(alldocs,total_examples=self.corpus_count)

	#model.save("/home/joao/Wiki2016/model/word2vec_gensim.txt")
    model.wv.save_word2vec_format("/home/joao/Wiki2016/model/word2vec/word2vec_gensim.min10words.20.words.skipgram.txt",binary=False)
    model.save("/home/joao/Wiki2016/model/word2vec/word2vec_gensim.min10words.20.words.skipgram")

    end = time()
	#print("Total procesing time: %d seconds") %(end - begin)