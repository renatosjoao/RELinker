#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:19:37 2019

@author: renato
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
import nerNgrammodule


WIKIPEDIA_URI_BASE = u"https://en.wikipedia.org/wiki/{}"

## This is going to the my main method
### It requires the input text
#   Then it processes the input text and returns recognized tokens by the stanford NER 
class SimilarityPrior():
        
    ''' 
    
    This method is meant to produce a list of mentions and the selected entity
    
    This setting combines the candidate entity prior with the cosine similarity  of the  tokens from the mention 
    and the tokens from the candidate entity 
    
    For example:
        
        Mention         Cand. entity                            CosSim          PRIOR         FINAL SCORE
        taliban         taliban                                 1.000000       0.4985          x
        taliban         islamic emirate of afghanistan          0.749854       0.7954   
        taliban         tehrik-i-taliban pakistan               0.671049       0.9854
    
    '''


    def cos(self,mentVecs,cEVecs):
        dot = np.dot(mentVecs,cEVecs)
        norma = np.linalg.norm(mentVecs)
        normb = np.linalg.norm(cEVecs)
        cosDist = dot / (norma * normb)
        return cosDist
    
    def normalize_title(self,title):
        elems = title.split()
        title = []
        for e in elems:
            if e[0] == "(":
                e = e[0] + e[1].upper() + e[2:]
            else:
                e = e[0].upper() + e[1:]
            
            title.append(e)

        title = ' '.join(title).replace(" ", "_")
        
        title = title.strip().replace(" ", "_")
        return title
    
    
    def disambiguate(self, tokensList, sentence, CandEntities, Embedding):      
        DisambiguationList = []
        mcObj = MentionContext.MentionContext()
        ecObj = EntityContext.EntityContext()
   
        for elem in tokensList : 
            [Nmention, pos]  = elem
            #print("{} {} ".format(Nmention,pos))
            mention = Nmention.lower()
            
            #tokens = mention.split()
            
            #if len(tokens) == 1 :
                
            #    try:
            #        mentVecs = Embedding.get_word_vector(mention)
            #    except Exception:
            #        mentVecs = zeros(500, dtype=np.float32)
            #        continue
            #else :
                #mentVecs = zeros(500, dtype=np.float32).reshape(-1, 500)
            mentVecs = zeros(500, dtype=np.float32)
            contador = 0.0
            
            tokens  = mcObj.getContextWindow(sentence,mention,pos,5) #win_size
            
            
            for t in tokens :
                try:
                    vec = Embedding.get_word_vector(t)
                    contador+=1.0
                    mentVecs += vec
                except Exception:
                        continue
                
                
            if contador != 0.0:
                mentVecs = mentVecs / contador
                
                
            if mention in CandEntities:
                candEList = CandEntities[mention]
                finalScore = 0.0
                slctEnt = ""
                for elem in candEList:
                    extract = ecObj.getWikiSection(elem)
                    tokens = []
                    pscore = 0.0
                    if extract == None:
                        ce, pscore = elem.split('\t')
                        pscore = np.float32(pscore)
                        tokens = ce.split()
                    else:
                        ce, pscore = elem.split('\t')
                        pscore = np.float32(pscore)
                        tokens = extract.split()
                    
                    if len(tokens) == 1 :
                        try:    
                            cEVecs = Embedding.get_word_vector(ce)
                            #cEVecs = Embedding.get_word_vector(candE)
                        except Exception:                 
                            cEVecs = zeros(500, dtype=np.float32)
                            continue
                    else :
                        cEVecs = zeros(500, dtype=np.float32)
                        #cEVecs = zeros(500, dtype=np.float32).reshape(-1, 500)
                    
                        contador = 0.0
                        for t in tokens :
                            try:
                                vec = Embedding.get_word_vector(t)
                                contador+=1.0
                                cEVecs += vec
                            except Exception:
                                continue
                        if contador != 0:
                            cEVecs = cEVecs / contador
                    dot = np.dot(mentVecs,cEVecs)
                    norma = np.linalg.norm(mentVecs)
                    normb = np.linalg.norm(cEVecs)
                    
                    if norma == 0.0 or normb == 0.0:
                        cosDist = 0.0
                    else:
                        cosDist = dot / (norma * normb)
                    
                    
                    #print("cosDist" + str(cosDist))
                    score =  (pscore + cosDist) /2.0
                    
                    
                    if score > finalScore :
                        slctEnt = ce
                        finalScore = score
                if slctEnt :
                    print("{} {} >> {} << ".format(Nmention,pos,slctEnt))
                    DisambiguationList.append(Nmention + "\t" + str(pos) + "\t" +  str(slctEnt) + "\t" + str(finalScore))
            else:
                slctEnt = "NIL"
                finalScore = 0.0 

            

        return DisambiguationList           

    def outText(self, DisambiguationList,raw_text):
        #print()
        #print(raw_text)
        #print()
        outputText = raw_text
        finalText = []
        previous = len(raw_text) 
        listSize = len(DisambiguationList)
        cont = 0
        for elem in reversed(DisambiguationList):
            tok, pos, Elink, score = elem.split('\t')
            
            #print("%s %s %s %s" %(tok,pos,Elink,score))
            
            pos = int(pos)
            if pos != -1:
            
                cont+=1
                if cont == listSize:
                    pos = 0
            #print(tok+"\t"+pos)
                substring = raw_text[pos:previous]
                #if tok.lower() in PriorModel:
                #    Etitle = PriorModel[tok.lower()]
                #else:
                #    Etitle = 'NIL'

                #if Elink!= 'NIL' :
                Elink = WIKIPEDIA_URI_BASE.format(self.normalize_title(Elink))
                outputText = substring.replace(tok,"<a href=\""+Elink+"\"  target=\"_blank\">"+tok+"</a>",1)
                #else:
                 #outputText = substring.replace(tok,"<a href=\"--NIL--\">"+tok+"</a>",1)
                #    outputText = substring.replace(tok,"<a>"+tok+"</a>",1)
                previous = pos
    #print(outputText)
                finalText.append(outputText)

        text = [str(lint) for lint in reversed(finalText)]
        text = "".join(text)
        return text

    
if __name__ == '__main__':
    inputText = "WASHINGTON (Reuters) - U.S. House Intelligence Committee Chairman Adam Schiff said on Monday he fully expected four White House officials scheduled for depositions with investigators in the House’s impeachment inquiry to defy congressional subpoenas. We expect the witnesses who have been subpoenaed to come in this afternoon, at White House instruction, also to be no-shows. This will only further add to the body of evidence on a potential obstruction of Congress charge against the president, Schiff told reporters. The officials were called in to testify in the House’s ongoing impeachment inquiry stemming from a July 25 call in which U.S. President Donald Trump pressed Ukrainian President Volodymyr Zelenskiy to investigate one of Trump’s domestic political rivals, former vice president and leading Democratic presidential candidate Joe Biden."
  
    LPDic = nerNgrammodule.loadLP()
    
    
    tokensList = nerNgrammodule.getTokensList(LPDic,inputText,5,0.05)
    #print(tokensList)
    
    Embedding = embeddingsmodule.loadEmbedding()
    
    CandEntities  = candentitiesmodule.loadCandidates(2)
    
    SP = SimilarityPrior()
    DisambiguationList = SP.disambiguate(tokensList,inputText, CandEntities, Embedding)
    print(DisambiguationList)
    print()
    outputContent = SP.outText(DisambiguationList,inputText)
    print(outputContent)    
    
    
    