#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:11:56 2019

@author: renato
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:43:37 2019

@author: renato
"""


import gensim
import os
import re
from gensim.models import Doc2Vec
from gensim.models.doc2vec import  TaggedDocument
import multiprocessing
from collections import OrderedDict
#import logging
import sys
from random import shuffle
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy import zeros




numlines = 0
numAnnGT = 0
numNIL = 0



from time import time

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)

def _remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
    return reg.sub(' ', string)

def _remove_stop_words(string,sw=[]):
    return ' '.join([word if word not in sw else '' \
                     for word in string.strip().split(' ')])

def _trim_string(string):
    # remove extra spaces, remove trailing spaces, lower the case
    return re.sub('\s+',' ',string).strip().lower()


def loadGT():
    GT = {}
    f = open("/home/joao/datasets/conll_corpus/conllYAGO_testb_GT_NONIL.tsv", 'r',1,encoding='utf-8')
    for i in f:
        doc, mention, offset, el = i.split('\t')
        key = doc+"\t"+mention.lower()+"\t"+offset
        val = el.lower()
        GT[key] = val
    return GT
##########################################################################################################################


##########################################################################################################################
def loadMappings():###Ambiverse mappings

    AmbiMap = {}
    BabMap = {}
    TagMap = {}
    SpotMap = {}

    ambi_mappings = open('/home/joao/datasets/conll_corpus/mappings/conll_ambiverse_testb.mappings','r',1,encoding='utf-8')
    for i in ambi_mappings:
        doc, mention, offset, lAMB = i.split('\t')
        key = doc+"\t"+mention.lower()+"\t"+offset
        lAMB = lAMB.replace("_"," ").lower().strip()
        AmbiMap[key] = lAMB

    babel_mappings = open('/home/joao/datasets/conll_corpus/mappings/conll_bfy_testb.mappings','r',1,encoding='utf-8')
    for i in babel_mappings:
        doc, mention, offset,lBAB = i.split('\t')
        key = doc+"\t"+mention.lower()+"\t"+offset
        lBAB = lBAB.replace("_"," ").lower().strip()
        BabMap[key] = lBAB

    spot_mappings = open('/home/joao/datasets/conll_corpus/mappings/conll_spotlight_testb.mappings','r',1,encoding='utf-8')
    for i in spot_mappings:
        doc, mention, offset, lSPOT = i.split('\t')
        key = doc+"\t"+mention.lower()+"\t"+offset
        lSPOT = lSPOT.replace("_"," ").lower().strip()
        SpotMap[key] = lSPOT
    tagme_mappings = open('/home/joao/datasets/conll_corpus/mappings/conll_tagme_testb.mappings','r',1,encoding='utf-8')

    for i in tagme_mappings:
        doc, mention, offset,lTAG = i.split('\t')
        key = doc+"\t"+mention.lower()+"\t"+offset
        lTAG = lTAG.replace("_"," ").lower().strip()
        TagMap[key] = lTAG

    return [AmbiMap,BabMap,TagMap,SpotMap]




def predict(d2v_model,GT,AmbiMap,BabMap,TagMap,SpotMap):
    cont = 0
    ZeroTool = 0
    OneTool = 0
    TwoTool = 0
    ThreeTool = 0
    FourTool = 0
    TP = 0
    numRECOGNIZED = 0
    for key,value in GT.items():
        GTLink = value.lower().strip()
        terms = key.split('\t')
        docID = terms[0].lower().strip()
        docVecGT = d2v_model.docvecs[docID].reshape(-1, 300)
    #docGT = docMap[docID]

    # mentions recognized by 0 EL tools
        if (key not in AmbiMap) and (key not in BabMap) and (key not in SpotMap) and (key not in TagMap) :
            ZeroTool += 1
            continue

    # mentions recognized by 1 EL tools --- Ambiverse
        if (key in AmbiMap) and (key not in BabMap) and (key not in SpotMap) and (key not in TagMap) :
            OneTool += 1
            numRECOGNIZED+=1
            Alink = AmbiMap[key].lower().strip()
            if Alink == GTLink :
                TP+=1
#        else:
#            out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
        #continue
     # mentions recognized by 1 EL tools --- Babelfy
        if (key not in AmbiMap) and (key in BabMap) and (key not in SpotMap) and (key not in TagMap) :
            OneTool += 1
            numRECOGNIZED+=1
            Blink = BabMap[key].lower().strip()
            if Blink == GTLink :
                TP+=1
#        else:
#            out.write(key+"\t"+GTLink+"\t"+Blink+"\n")
        #continue
    # mentions recognized by 1 EL tools --- Spotlight
        if (key not in AmbiMap) and (key not in BabMap) and (key in SpotMap) and (key not in TagMap) :
            OneTool += 1
            numRECOGNIZED+=1
            Slink = SpotMap[key].lower().strip()
            if Slink == GTLink :
                TP+=1
#        else:
#            out.write(key+"\t"+GTLink+"\t"+Slink+"\n")
        #continue
    # mentions recognized by 1 EL tools --- Tagme
        if (key not in AmbiMap) and (key not in BabMap) and (key not in SpotMap) and (key in TagMap) :
            OneTool += 1
            numRECOGNIZED+=1
            Tlink = TagMap[key].lower().strip()
            if Tlink == GTLink :
                TP+=1
#        else:
#            out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
        #continue
    # mentions recognized by 2 EL tools --- Ambiverse and Babelfy
        if (key in AmbiMap) and (key in BabMap) and (key not in SpotMap) and (key not in TagMap) :
            TwoTool += 1
            numRECOGNIZED+=1
            Alink = AmbiMap[key].lower().strip()
            Blink = BabMap[key].lower().strip()
            if Alink == Blink:
                if Alink == GTLink:
                    TP+=1
            #continue
            else:
            #Predict using word2vec model
                docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                try:
                      docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                except Exception:
                      docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                try:
                      docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                except Exception:
                      docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
        #
                scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                scoreA = scoreA[0][0]
                scoreB = cosine_similarity(docVecGT, docVecB)
                scoreB = scoreB[0][0]
            #u1 = modelloaded.infer_vector(u)
            #v1 = modelloaded.infer_vector(v)
            #dist=scipy.spatial.distance.cosine(u1, v1)
                if scoreA > scoreB :
                    if Alink == GTLink :
                        TP+=1
                    #print(str(scoreA))
#                    else:
#                       out.write(key+"\t"+GTLink+">"+str(pic)+"\t"+Alink+":"+str(scoreA)+"\t"+Blink+":"+str(scoreB)+"\n")
                   #out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                else:
                    if Blink == GTLink :
                        TP+=1
                   # print(str(scoreB))
    #                else:
    #                    out.write(key+"\t"+GTLink+">"+str(pic)+"\t"+Alink+":"+str(scoreA)+"\t"+Blink+":"+str(scoreB)+"\n")
                    #out.write(key+"\t"+GTLink+"\t"+Blink+"\n")

        #averaging words
        #docA = WikiMap[Alink]
        #docB = WikiMap[Blink]
        #scoreA = d2v_model.n_similarity(docGT, docA)
        #scoreB = d2v_model.n_similarity(docGT, docB)
        #continue
# =============================================================================
     # mentions recognized by 2 EL tools --- Ambiverse and Spotlight
        if (key in AmbiMap) and (key not in BabMap) and (key in SpotMap) and (key not in TagMap) :
             TwoTool += 1
             numRECOGNIZED+=1
             Alink = AmbiMap[key].lower().strip()
             Slink = SpotMap[key].lower().strip()
             if Alink == Slink:
                 if Alink == GTLink:
                     TP+=1
                 #continue
             else:
                   #Predict using word2vec model
                 docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                 except Exception:
                     docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)

                 scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                 scoreA = scoreA[0][0]
                 scoreS = cosine_similarity(docVecGT, docVecS)
                 scoreS = scoreS[0][0]
                 if scoreA > scoreS :
                     if Alink == GTLink :
                         TP+=1
                 else:
                     if Slink == GTLink :
                         TP+=1
                 #continue
      # mentions recognized by 2 EL tools --- Ambiverse and Tagme
        if (key in AmbiMap) and (key not in BabMap) and (key not in SpotMap) and (key in TagMap) :
              TwoTool += 1
              numRECOGNIZED+=1
              Alink = AmbiMap[key].lower().strip()
              Tlink = TagMap[key].lower().strip()
              if Alink == Tlink:
                  if Alink == GTLink:
                      TP+=1
              #continue
              else:
                  #Predict using word2vec model
                  docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                  docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                  try:
                      docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                  except Exception:
                      docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                  try:
                      docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                  except Exception:
                      docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)
                  scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                  scoreA = scoreA[0][0]
                  scoreT = cosine_similarity(docVecGT, docVecT)
                  scoreT = scoreT[0][0]

                  if scoreA > scoreT :
                      if Alink == GTLink :
                          TP+=1
     #            else:
 #                out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                  else:
                      if Tlink == GTLink :
                          TP+=1
                #else:
 #                    out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
          #continue

      # mentions recognized by 2 EL tools --- Babelfy and Spotlight
        if (key not in AmbiMap) and (key in BabMap) and (key in SpotMap) and (key not in TagMap) :
             TwoTool += 1
             numRECOGNIZED+=1
             Blink = BabMap[key].lower().strip()
             Slink = SpotMap[key].lower().strip()
             if Blink == Slink:
                 if Blink == GTLink:
                     TP+=1
                 #continue
             else:
                 #Predict using word2vec model
                 docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                 except Exception:
                     docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)
                 scoreB = cosine_similarity(docVecGT, docVecB)  # this is actually doc2vec
                 scoreB = scoreB[0][0]
                 scoreS = cosine_similarity(docVecGT, docVecS)
                 scoreS = scoreS[0][0]
                 if scoreB > scoreS :
                     if Blink == GTLink :
                         TP+=1
        #             else:
 #                 out.write(key+"\t"+GTLink+"\t"+Blink+"\n")
                 else:
                     if Slink == GTLink :
                         TP+=1
 #             else:
 #                 out.write(key+"\t"+GTLink+"\t"+Slink+"\n")
  #        continue
#     # mentions recognized by 2 EL tools --- Babelfy and Tagme
        if (key not in AmbiMap) and (key in BabMap) and (key not in SpotMap) and (key in TagMap) :
             TwoTool += 1
             numRECOGNIZED+=1
             Blink = BabMap[key].lower().strip()
             Tlink = TagMap[key].lower().strip()
             if Blink == Tlink:
                 if Blink == GTLink:
                     TP+=1
              #continue
             else:
             #Predict using word2vec model
                 docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                 except Exception:
                     docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
                 try:
                     docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                 except Exception:
                     docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)

                 scoreB = cosine_similarity(docVecGT, docVecB)  # this is actually doc2vec
                 scoreB = scoreB[0][0]
                 scoreT = cosine_similarity(docVecGT, docVecT)
                 scoreT = scoreT[0][0]
                 if scoreB > scoreT :
                     if Blink == GTLink :
                         TP+=1
             #    else:
         #        out.write(key+"\t"+GTLink+"\t"+Blink+"\n")
                 else:
                     if Tlink == GTLink :
                         TP+=1
             #    else:
         #        out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
          #continue
      # mentions recognized by 2 EL tools --- Spotlight and Tagme
        if (key not in AmbiMap) and (key not in BabMap) and (key in SpotMap) and (key in TagMap) :
             TwoTool += 1
             numRECOGNIZED+=1
             Slink = SpotMap[key].lower().strip()
             Tlink = TagMap[key].lower().strip()
             if Slink == Tlink:
                 if Slink == GTLink:
                     TP+=1
               #continue
             else:#Predict using word2vec model
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)
                 try:
                     docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                 except Exception:
                     docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)
                 scoreS = cosine_similarity(docVecGT, docVecS)  # this is actually doc2vec
                 scoreS = scoreS[0][0]
                 scoreT = cosine_similarity(docVecGT, docVecT)
                 scoreT = scoreT[0][0]
                 if scoreS > scoreT :
                     if Slink == GTLink :
                        TP+=1
                  # else:
              #       out.write(key+"\t"+GTLink+"\t"+Slink+"\n")
                 else:
                     if Tlink == GTLink :
                             TP+=1
                   #else:
                   #      out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
         #continue
     # mentions recognized by 3 EL tools --- Ambiverse Babelfy Spotligh
        if (key in AmbiMap) and (key in BabMap) and (key in SpotMap) and (key not in TagMap) :
             ThreeTool += 1
             numRECOGNIZED+=1
             Alink = AmbiMap[key].lower().strip()
             Blink = BabMap[key].lower().strip()
             Slink = SpotMap[key].lower().strip()
             if (Alink == Blink) and (Alink == Slink) :
                 if Alink == GTLink:
                     TP+=1
                 #continue
             else:#Predict using word2vec model
                 docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                 except Exception:
                     docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                 try:
                     docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                 except Exception:
                     docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)
                 scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                 scoreA = scoreA[0][0]
                 scoreB = cosine_similarity(docVecGT, docVecB)  # this is actually doc2vec
                 scoreB = scoreB[0][0]
                 scoreS = cosine_similarity(docVecGT, docVecS)
                 scoreS = scoreS[0][0]
                 if (scoreA > scoreB ) and (scoreA > scoreS )  :
                     if Alink == GTLink :
                         TP+=1
                     #else:
                 #    out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                 if (scoreB > scoreA ) and (scoreB > scoreS )  :
                     if Blink == GTLink :
                         TP+=1
                  #else:
                 #    out.write(key+"\t"+GTLink+"\t"+Blink+"\n")
                 if (scoreS > scoreA ) and (scoreS > scoreB )  :
                     if Slink == GTLink :
                         TP+=1
                 #else:
                 #    out.write(key+"\t"+GTLink+"\t"+Slink+"\n")
             #continue
     #
         # mentins recognized by 3 EL tools --- Ambiverse Babelfy and Tagme
        if (key in AmbiMap) and (key in BabMap) and (key not in SpotMap) and (key in TagMap) :
             ThreeTool += 1
             numRECOGNIZED+=1
             Alink = AmbiMap[key].lower().strip()
             Blink = BabMap[key].lower().strip()
             Tlink = TagMap[key].lower().strip()
             if (Alink == Blink) and (Alink == Tlink) :
                 if Alink == GTLink:
                     TP+=1
                 #continue
             else:#Predict using word2vec model
                 docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                   #docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                 except Exception:
                     docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                 try:
                     docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                 except Exception:
                     docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
                 try:
                     docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                 except Exception:
                     docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)

                 scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                 scoreA = scoreA[0][0]
                 scoreB = cosine_similarity(docVecGT, docVecB)  # this is actually doc2vec
                 scoreB = scoreB[0][0]
                 scoreT = cosine_similarity(docVecGT, docVecT)
                 scoreT = scoreT[0][0]

                 if (scoreA > scoreB ) and (scoreA > scoreT )  :
                     if Alink == GTLink :
                         TP+=1
                 #else:
                 #    out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                 if (scoreB > scoreA ) and (scoreB > scoreT )  :
                     if Blink == GTLink :
                         TP+=1
                 #else:
                 #    out.write(key+"\t"+GTLink+"\t"+Blink+"\n")
                 if (scoreT > scoreA ) and (scoreT > scoreB )  :
                     if Tlink == GTLink :
                       TP+=1
                 #else:
                 #    out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
               #continue
         # mentions recognized by 3 EL tools --- Ambiverse SpotLight and Tagme
        if (key in AmbiMap) and (key not in BabMap) and (key in SpotMap) and (key in TagMap) :
             ThreeTool += 1
             numRECOGNIZED+=1
             Alink = AmbiMap[key].lower().strip()
             Slink = SpotMap[key].lower().strip()
             Tlink = TagMap[key].lower().strip()
             if (Alink == Slink) and (Alink == Tlink) :
                 if Alink == GTLink:
                     TP+=1
                 #continue
             else:
                 #Predict using word2vec model
                 docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                 except Exception:
                     docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)
                 try:
                     docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                 except Exception:
                     docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)
                 scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                 scoreA = scoreA[0][0]
                 scoreS = cosine_similarity(docVecGT, docVecS)  # this is actually doc2vec
                 scoreS = scoreS[0][0]
                 scoreT = cosine_similarity(docVecGT, docVecT)
                 scoreT = scoreT[0][0]
                 if (scoreA > scoreS ) and (scoreA > scoreT )  :
                     if Alink == GTLink :
                         TP+=1
                   #else:
                       #    out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                 if (scoreS > scoreA ) and (scoreS > scoreT )  :
                     if Slink == GTLink :
                         TP+=1
                   #else:
                   #    out.write(key+"\t"+GTLink+"\t"+Slink+"\n")
                 if (scoreT > scoreA ) and (scoreT > scoreS )  :
                     if Tlink == GTLink :
                         TP+=1
                       #else:
                       #    out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
             #continue

             # mentions recognized by 3 EL tools --- Babelfy SpotLight and Tagme
        if (key not in AmbiMap) and (key in BabMap) and (key in SpotMap) and (key in TagMap) :
             ThreeTool += 1
             numRECOGNIZED+=1
             Blink = BabMap[key].lower().strip()
             Slink = SpotMap[key].lower().strip()
             Tlink = TagMap[key].lower().strip()
             if (Blink == Slink) and (Blink == Tlink) :
                 if Blink == GTLink:
                     TP+=1
                     #continue
             else:
                 #Predict using word2vec model
                 docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                 except Exception:
                     docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)
                 try:
                     docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                 except Exception:
                     docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)
                 scoreB = cosine_similarity(docVecGT, docVecB)
                 scoreB = scoreB[0][0]
                 scoreS = cosine_similarity(docVecGT, docVecS)  # this is actually doc2vec
                 scoreS = scoreS[0][0]
                 scoreT = cosine_similarity(docVecGT, docVecT)
                 scoreT = scoreT[0][0]
     #
                 if (scoreB > scoreS ) and (scoreB > scoreT )  :
                     if Blink == GTLink :
                        TP+=1
                 if (scoreS > scoreB ) and (scoreS > scoreT )  :
                     if Slink == GTLink :
                        TP+=1
                 if (scoreT > scoreB ) and (scoreT > scoreS )  :
                     if Tlink == GTLink :
                        TP+=1
         # mentions recognized by 4 EL tools
        if (key in AmbiMap) and (key in BabMap) and (key in SpotMap) and (key in TagMap) :
            FourTool += 1
            numRECOGNIZED+=1
            Alink = AmbiMap[key].lower().strip()
            Blink = BabMap[key].lower().strip()
            Slink = SpotMap[key].lower().strip()
            Tlink = TagMap[key].lower().strip()
            if (Alink == Blink) and (Alink == Slink) and (Alink == Tlink) :
                if Alink == GTLink:
                    TP+=1
                #else:
                    #out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                 #continue

            else:
                 docVecA = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecB = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecS = zeros(300, dtype=np.float32).reshape(-1, 300)
                 docVecT = zeros(300, dtype=np.float32).reshape(-1, 300)
                 try:
                     docVecA = d2v_model.docvecs[Alink].reshape(-1, 300)
                 except Exception:
                     docVecA =  d2v_model.infer_vector(Alink).reshape(-1, 300)
                 try:
                     docVecB = d2v_model.docvecs[Blink].reshape(-1, 300)
                 except Exception:
                     docVecB =  d2v_model.infer_vector(Blink).reshape(-1, 300)
                 try:
                     docVecS = d2v_model.docvecs[Slink].reshape(-1, 300)
                 except Exception:
                     docVecS =  d2v_model.infer_vector(Slink).reshape(-1, 300)
                 try:
                     docVecT = d2v_model.docvecs[Tlink].reshape(-1, 300)
                 except Exception:
                     docVecT =  d2v_model.infer_vector(Tlink).reshape(-1, 300)

                 scoreA = cosine_similarity(docVecGT, docVecA)  # this is actually doc2vec
                 scoreA = scoreA[0][0]
                 scoreB = cosine_similarity(docVecGT, docVecB)
                 scoreB = scoreB[0][0]
                 scoreS = cosine_similarity(docVecGT, docVecS)  # this is actually doc2vec
                 scoreS = scoreS[0][0]
                 scoreT = cosine_similarity(docVecGT, docVecT)
                 scoreT = scoreT[0][0]
                 if (scoreA > scoreB ) and (scoreA > scoreS )  and (scoreA > scoreT ):
                     if Alink == GTLink :
                         TP+=1
                     #else:
                     #    out.write(key+"\t"+GTLink+"\t"+Alink+"\n")
                 if (scoreB > scoreA ) and (scoreB > scoreS )  and (scoreB > scoreT ):
                     if Blink == GTLink :
                         TP+=1
                     #else:
                     #    out.write(key+"\t"+GTLink+"\t"+Blink+"\n")
                 if (scoreS > scoreA ) and (scoreS > scoreB )  and (scoreS > scoreT ):
                     if Slink == GTLink :
                         TP+=1
                     #else:
                     #    out.write(key+"\t"+GTLink+"\t"+Slink+"\n")
                 if (scoreT > scoreA ) and (scoreT > scoreB )  and (scoreT > scoreS ):
                     if Tlink == GTLink :
                         TP+=1
                     #else:
                     #    out.write(key+"\t"+GTLink+"\t"+Tlink+"\n")
        cont+=1


    P = 0.0;
    R = 0.0;
    F = 0.0;

    P =  TP / numRECOGNIZED
    R =  TP / len(GT)
    F = 2*((P*R)/(P+R))
    return [P,R,F]
#print("Meta EL Prediction");
#print(" *********************** ***********************");
#print("P:"+ str(P));
#print("R:"+ str(R));
#print("F:"+ str(F));

if __name__ == '__main__':
      program = os.path.basename(sys.argv[0])
      #logger = logging.getLogger(program)

      #logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
      #logging.root.setLevel(level=logging.INFO)
      #logger.info("running %s" % ' '.join(sys.argv))


      GT = loadGT()

      AmbiMap, BabMap,TagMap,SpotMap = loadMappings()
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
      #print("done loading data set.")


      begin = time()

      topTitles = set()
      wiki_top = open('/home/joao/Wiki2016/mentions.50.txt', 'r',1,encoding='utf-8')

      for linha in wiki_top:
          linha = linha.strip()
          topTitles.add(linha)
      #print("Num of titles "+str(len(topTitles)))

      wikiMAP = {}
      wiki_f = open('/home/joao/Wiki2016/wiki.en.tsv', 'r',1,encoding='utf-8')
      for line in wiki_f:
          splitline = line.split('\t')
          title = splitline[0]
          conteudo = splitline[1]
          wikiMAP[title] = conteudo



      for par_size in 1000,1500,2000,2500,3000 :
          for win in 3,5,10,15,20,25,40,50:
              alldocs = []
              for t,c in docMap.items():
                  sentences = c.split()
                  alldocs.append(TaggedDocument(sentences, [t]))


              for title in topTitles:
                  try :
                      conteudo = wikiMAP[title]
                      sentences = conteudo.split()
                      if len(sentences) >= par_size:
                          sentences = sentences[:par_size]
                          alldocs.append(TaggedDocument(sentences, [title]))
                      else:
                          alldocs.append(TaggedDocument(sentences, [title]))
                  except Exception:
                        continue






              cores = multiprocessing.cpu_count()
              assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

              cores = multiprocessing.cpu_count()
# =============================================================================
#     model = Doc2Vec(size=300, window=10, min_count=10, alpha=0.05, min_alpha=0.025, workers=16,dm=0, negative=5, sample=0 )
# =============================================================================

      #PV-DBOW plain
              PV_DBOW =  Doc2Vec(dm=0, size=300, window=5, negative=5, hs=0, min_count=5, sample=0, workers=18)
      #PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
      #        PV_DM = Doc2Vec(dm=1, size=300, window=win, negative=5, hs=0, min_count=5, sample=0, workers=20 ,alpha=0.01)
      #PV-DM w/ concatenation - big, slow, experimental mode
      #window=5 (both sides) approximates paper's apparent 10-word total window size
      #PV_DM_CONCAT = Doc2Vec(dm=1, dm_concat=1, size=300, window=5, negative=5, hs=0, min_count=10, sample=0, workers=18)



              PV_DBOW.build_vocab(alldocs)
      #print("%s vocabulary scanned & state initialized PV_DBOW")
              #PV_DM.build_vocab(alldocs)
      #print("%s vocabulary scanned & state initialized PV_DM")
      #PV_DM_CONCAT.build_vocab(alldocs)
      #print("%s vocabulary scanned & state initialized PV_DM_CONCAT")

      #print("Training PV_DBOW")
              PV_DBOW.train(alldocs, total_examples=PV_DBOW.corpus_count, epochs=500)
      #print("Training PV_DM")
              #PV_DM.train(alldocs, total_examples=PV_DM.corpus_count,epochs=500)
              P,R,F = predict(PV_DBOW,GT,AmbiMap,BabMap,TagMap,SpotMap)
              print("PV_DBOW Par_size:"+str(par_size)+"\twin_size:"+str(win)+ "\tP:"+str(P)+ "\tR:"+str(R)+ "\tF:"+str(F))


      #print("Training PV_DM_CONCAT")
      #PV_DM_CONCAT.train(alldocs, total_examples=PV_DM_CONCAT.corpus_count, epochs=500)


#      models_by_name = OrderedDict((str(PV_DBOW), PV_DBOW) for model in simple_models)

      #model_dbow_dmm = ConcatenatedDoc2Vec([PV_DBOW,PV_DM])
      #model_dbow_dmc = ConcatenatedDoc2Vec([PV_DBOW, PV_DM_CONCAT])





      #PV_DBOW.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.PV_DBOW.min10words.20words.top50")
      #PV_DM.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.PV_DM.min5words.200words.top50")
      #PV_DM_CONCAT.save("/home/joao/Wiki2016/model/doc2vec/doc2vec_gensim.PV_DM_CONCAT.min10words.200words.top50")


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
