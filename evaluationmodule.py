#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:00:54 2019

@author: renato
"""


''' 

    This is meant to be the evaluation file
    It requires the produced mappings from entity linking and the GT mappings

'''

import os
import nerNgrammodule
import gtmodule
import embeddingsmodule
import candentitiesmodule
from numpy import zeros
import MentionContext
import EntityContext
import MethodSimilarity


#os.chdir("/home/renato/datasets/aquaint/RawTexts")
#for file in glob.glob("*.htm"):
#    print(file)
    
class Evaluation:
    
    
    #def evaluate(self, GroundT, ):
         #fin = open("./resultsEL/iitb.sim", 'w',1,encoding='utf-8')
        
        
    def evalEntityRec(self):
        GT = gtmodule.Groundtruth()
        _gtDIC = GT.loadGT("conll")
        
        #print(_gtDIC)
        _erLIST = self.loadEntityRec("conll")
        TP = 0
        for element in _erLIST:
            [doc,mention,offset] = element
            key = doc.strip()+"\t"+mention.strip()+"\t"+offset.strip()
            #print(key)
            if key in _gtDIC:
                TP+=1
            
        
        _totElem = len(_erLIST)
        _totGT = len(_gtDIC)
        #print(_tolGT)
        P = float(TP/_totElem)
        P = float(P)
        R = float(TP/_totGT)
        R = float(R)
        F = float(2*((P*R)/(P+R)) )
        F = float(F)
        #print("{}".format(_totElem))
        print("P:{}  R:{}  F:{}".format(P,R,F))
        #print("P:{}  R:{}".format(P,R))
            
        
        
    def loadEntityRec(self,corpus):    
        _erDIC = []
        
        GT = gtmodule.Groundtruth()
        _gtDIC = GT.loadGT("conll")
        
        #for i in [0.005, 0.01, 0.05, 0.1, 0.5, 0.8]:
        #fout = open("./resultsER/msnbc."+str(i)+".er", 'w',1,encoding='utf-8')
        fread = open("./resultsER/conll.0.5.er", 'r',1,encoding='utf-8')
        for i in fread:
            i = i.lower()
            #numlines+=1
            doc, mention, offset = i.split('\t')
            
            key = [doc,mention.lower(),offset]
            
            _erDIC.append(key)
        fread.close()
        
        return _erDIC

 
        
        
        
    def dumpEntityRec(self):
        #print(np.linspace(0,0.9,10))
        corpus = "iitb"
    
        #"/home/renato/datasets/conll/TEXT_FILES"
        #"/home/renato/datasets/msnbc/RawTexts"
        #"/home/renato/datasets/ace2004/TEXT_FILES"
        #"/home/renato/datasets/iitb/TEXT_FILES"

        LPDic = nerNgrammodule.loadLP()
        for i in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8]:
    
            fout = open("./resultsER/conll."+str(i)+".er", 'w',1,encoding='utf-8')
    
            for root, dirs, files in os.walk("/home/joao/datasets/conll/TEXT_FILES"):
                for file in files:
                #if file.endswith("txt"):
                    full_file_path = os.path.join(root, file)
                    f = open(full_file_path, 'r',1,encoding='utf-8')  #windows-1252 for MSNBC
                    raw_text =f.read()
                    print(full_file_path)
                    #print(raw_text)
                    doc = full_file_path.split("/")
                    docid = doc[-1]
              
                    tokensList = nerNgrammodule.getTokensList(LPDic,raw_text,5,float(i))
            
                    for token,pos in tokensList:
                        fout.write("{}\t{}\t{}\n".format(docid,token,pos))

                    f.close()
            fout.flush()
            fout.close()
        




    
if __name__ == '__main__':
    Ev = Evaluation()
    #Ev.dumpEntityRec()
    #Ev.evalEntityRec()
    
    GT = gtmodule.Groundtruth()
    _tokensDic = GT.loadGTterms("conll")
    _textDic  = GT.loadTextCorpus("conll")
    _gtDic   = GT.loadGT("conll")
    
    
    Embedding = embeddingsmodule.loadEmbedding()
    CandEntities  = candentitiesmodule.loadCandidates(20)
    
    S = MethodSimilarity.Similarity()
    
    fout = open("./resultsEL/conll.sim", 'w',1,encoding='utf-8')
    
    TP = 0.0
    #print(_textDic)
    _total = float(len(_gtDic.items()))
    print(_total)
    for _docid,tokensList in _tokensDic.items():
        inputText = _textDic[_docid]
    #   #print()
   #    print(_docid)
        DisambiguationList = S.disambiguate(tokensList, inputText,CandEntities, Embedding)
        #print(DisambiguationList)
        for eachItem in DisambiguationList:
            mention,pos,entity,score = eachItem.split("\t")
            
            key = _docid.lower().strip()+"\t"+mention.lower().strip()+"\t"+pos
            el = _gtDic[key]
            #print("{} => {}".format(el, entity))
            if el.lower() == entity.lower():
                TP+=1.0
            fout.write("{}\t{}\t{}\t{}\t{}\n".format(_docid,mention,int(pos),entity,score))
        
    fout.flush()
    fout.close()
    
    print(TP)
    print(_total)
    acc = TP/_total
    print("{}".format(acc))
            
            
            
