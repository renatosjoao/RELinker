#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:53:35 2019

@author: renato
"""

import csv
import nerNgrammodule
from html.parser import HTMLParser
import candentitiesmodule#from flask import g


WIKIPEDIA_URI_BASE = u"https://en.wikipedia.org/wiki/{}"
#outputText = outputText.replace(mention,"<a href=\"https://en.wikipedia.org/wiki/"+processedEntity+"\">"+mention+"</a>");
HTML_PARSER = HTMLParser()


class Priormodel():    
    
    def load_model(self):
        PriorModel = {}
        #CSVfile  = '/home/renato/eclipse-workspace/REntityLinking/Wiki2018/mentionentity.20181020.prior.csv'
        CSVfile  = './terms.prior.csv'

        with open(CSVfile, mode='r')  as csv_file:
            #csv_reader = csv.reader(csv_file, delimiter=',',quotechar='\'')
            csv_reader = csv.reader(csv_file,delimiter=',', quoting=csv.QUOTE_ALL)

            line_count = 0
            for row in csv_reader:
                line_count += 1
                    #print(row[0])
                if  row[0] in PriorModel:
                    pass
                else:
                    PriorModel[row[0]] = row[1]
                        #if line_count==20:
                        #    break
            print(f'Processed {line_count} lines.')
            #print(len(PriorModel))
            return PriorModel

    def title_to_uri(self,entity_title):
        '''
        Get the URI of the page describing a Wikipedia entity.
        :param entity_title: an entity title.
        :param lang: the Wikipedia language.
        '''
        return WIKIPEDIA_URI_BASE.format(self.normalize_title(entity_title))

    def normalize_title(self,title):
        '''
        Normalize a title to Wikipedia format. E.g. "barack obama" becomes "Barack_Obama"
        :param title: a title to normalize.
        '''
        elems = title.split()
        title = []
        for e in elems:
            e = e[0].upper() + e[1:]
            title.append(e)

        title = ' '.join(title).replace(" ", "_")
        
        title = title.strip().replace(" ", "_")
        return title

    def wiki_title(self,title):
        '''
        Given a normalized title, get the page title. E.g. "Barack_Obama" becomes "Barack Obama"
        :param title: a wikipedia title.
        '''
        return HTML_PARSER.unescape(title.strip(" _").replace("_", " "))


    def disambiguate(self, tokensList, CandEntities):
        DisambiguationList = []
        for elem in tokensList : 
            [Nmention, pos]  = elem
            mention = Nmention.lower()
            #print(len(CandEntities))
            if mention in CandEntities:
                candEList = CandEntities[mention]
                print(candEList)
                topK = candEList[0]
                
                slctEnt, pscore = topK.split('\t')
            else:
                slctEnt = "NIL"
                pscore = 0.0 
            
            DisambiguationList.append(Nmention + "\t" + str(pos) + "\t" +  str(slctEnt)+ "\t" + str(pscore))
                
                
        return DisambiguationList
    
        
    def outText(self, DisambiguationList, raw_text):
        print()
        outputText = raw_text
        finalText = []
        previous = len(raw_text) 
        listSize = len(DisambiguationList)
        cont = 0
        for elem in reversed(DisambiguationList):
            tok, pos, Elink, score = elem.split('\t')
            #print(Elink)
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

                #if Etitle!= 'NIL' : 
                Elink = WIKIPEDIA_URI_BASE.format(self.normalize_title(Elink))
                outputText = substring.replace(tok,"<a href=\""+Elink+"\"  target=\"_blank\">"+tok+"</a>",1)
                   
                #else:
             #outputText = substring.replace(tok,"<a href=\"--NIL--\">"+tok+"</a>",1)
               #     outputText = substring.replace(tok,"<a>"+tok+"</a>",1)
                previous = pos
    #print(outputText)
                finalText.append(outputText)

        text = [str(lint) for lint in reversed(finalText)]
        text = "".join(text)
        return text

    
        
if __name__ == "__main__":
    inputText = "WASHINGTON (Reuters) - U.S. House Intelligence Committee Chairman Adam Schiff said on Monday he fully expected four White House officials scheduled for depositions with investigators in the House’s impeachment inquiry to defy congressional subpoenas. We expect the witnesses who have been subpoenaed to come in this afternoon, at White House instruction, also to be no-shows. This will only further add to the body of evidence on a potential obstruction of Congress charge against the president, Schiff told reporters. The officials were called in to testify in the House’s ongoing impeachment inquiry stemming from a July 25 call in which U.S. President Donald Trump pressed Ukrainian President Volodymyr Zelenskiy to investigate one of Trump’s domestic political rivals, former vice president and leading Democratic presidential candidate Joe Biden."
    
    print()
    P = Priormodel() 
    LPDic = nerNgrammodule.loadLP()
    
    CandEntities  = candentitiesmodule.loadCandidates(2)
    tokensList = nerNgrammodule.getTokensList(LPDic,inputText,5,0.05)
    print(tokensList)
    DisambiguationList = P.disambiguate(tokensList,CandEntities)
    print()
    
    outputContent =  P.outText(DisambiguationList,inputText)
    print(outputContent)
    
    
    
    