#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:19:37 2019

@author: renato
"""

import numpy as np
import nerNgrammodule
import embeddingsmodule
import candentitiesmodule
from numpy import zeros
import MentionContext
import EntityContext


WIKIPEDIA_URI_BASE = u"https://en.wikipedia.org/wiki/{}"

## This is going to the my main method
### It requires the input text
#   Then it processes the input text and returns recognized tokens by the stanford NER 
class Similarity():
    ''' 
    
    This method is meant to produce a list of mentions and the selected entity
    
    The first setting simply compares word embeddings of the tokens from the mention 
    and the tokens from the candidate entity 
    
    For example:
        
        Mention         Cand. entity                            Score
        taliban         taliban                                 1.000000
        taliban         islamic emirate of afghanistan          0.749854
        taliban         tehrik-i-taliban pakistan               0.671049
    
    '''


    def cos(self, mentVecs,cEVecs):
        '''
        

        Parameters
        ----------
        mentVecs : numpy matrix --- dtype=np.float32
            Vector of the word embedding for the mention
        cEVecs : numpy matrix --- dtype=np.float32
            Vector of the word embedding for the candidate entity

        Returns
        -------
        cosDist : float
            Returns the cosine similarity value 

        '''
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
    
    
    

    def disambiguate(self, tokensList,sentence, CandEntities, Embedding): 
        DisambiguationList = []
        mcObj = MentionContext.MentionContext()
        ecObj = EntityContext.EntityContext()
        
        for elem in tokensList : 
            [Nmention, pos]  = elem
            #print("{} {} ".format(Nmention,pos))
            mention = Nmention.lower()
            '''
            tokens = mention.split()
            
            if len(tokens) == 1 :
                try:
                    mentVecs = Embedding.get_word_vector(mention)
                except Exception:
                    mentVecs = zeros(500, dtype=np.float32)
                    continue
            else :
                #mentVecs = zeros(500, dtype=np.float32).reshape(-1, 500)
                
              '''
            mentVecs = zeros(100, dtype=np.float32)
            contador = 0.0
            
            tokens  = mcObj.getContextWindow(sentence,mention,pos,5) #win_size

            #print("{} {}".format(mention,tokens))    
            
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
                #print()
                #print(candEList)
                #print()
                finalScore = 0.0
                slctEnt = ""
                 
                 
                
                for elem in candEList:
                    
                    #extract = ""#print("ELEM : {} ".format(elem))#EC = EntityContext(e)
                    #if elem in EntitiesContext:
                    #    extract = EntitiesContext[elem]
                        
                    #else:
                    #    extract = ecObj.getWikiSection(elem)
                    #    if len(extract) > 100:
                    #        fwriter.writerow((elem,extract))
                    tokens = []
                    #if extract == None:
                    ce, pscore = elem.split('\t')
                    tokens = ce.split()
                    #else:
                    #    ce, pscore = elem.split('\t')
                    #    tokens = extract.split()
                    
                    
                    if len(tokens) == 1 :
                        try:
                            cEVecs = Embedding.get_word_vector(ce)
                        except Exception:
                        
                            cEVecs = zeros(100, dtype=np.float32)
                            continue
                    else :
                        cEVecs = zeros(100, dtype=np.float32)
                        #cEVecs = zeros(500, dtype=np.float32).reshape(-1, 500)
                        contador = 0.0
                        for t in tokens :
                            try:
                                vec = Embedding.get_word_vector(t)
                                contador+=1.0
                                cEVecs += vec
                            except Exception:
                                continue
                        if contador != 0 :
                            cEVecs = cEVecs / contador
                            
                    dot = np.dot(mentVecs,cEVecs)
                    
                    norma = np.linalg.norm(mentVecs)
                    normb = np.linalg.norm(cEVecs)
                    if norma == 0.0 or normb == 0.0:
                        cosDist = 0.0
                    else:
                        cosDist = dot / (norma * normb)
                    
                    if cosDist > finalScore :
                        slctEnt = ce
                        finalScore = cosDist
                if slctEnt:
                    #print("{} {} >> {} << ".format(Nmention,pos,slctEnt))
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

    
    
    #https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles=Barack_Obama
    '''
    {
     batchcomplete: "",
     query: {
         normalized: [
             {
                 from: "Barack_Obama",
                 to: "Barack Obama"
                 }
             ],
         pages: {
             534366: {
                 pageid: 534366,
                 ns: 0,
                 title: "Barack Obama",
                 extract: "Barack Hussein Obama II ( (listen); born August 4, 1961) is an American attorney and politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American to be elected to the presidency. He previously served as a U.S. senator from Illinois from 2005 to 2008 and an Illinois state senator from 1997 to 2004. Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. After graduating, he became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. He represented the 13th district for three terms in the Illinois Senate from 1997 until 2004, when he ran for the U.S. Senate. He received national attention in 2004 with his March primary win, his well-received July Democratic National Convention keynote address, and his landslide November election to the Senate. In 2008, he was nominated for president a year after his campaign began, after a close primary campaign against Hillary Clinton. He was elected over Republican John McCain and was inaugurated on January 20, 2009. Nine months later, he was named the 2009 Nobel Peace Prize laureate. Regarded as a centrist New Democrat, Obama signed many landmark bills into law during his first two years in office. The main reforms that were passed include the Patient Protection and Affordable Care Act (commonly referred to as the "Affordable Care Act" or "Obamacare"), the Dodd–Frank Wall Street Reform and Consumer Protection Act, and the Don't Ask, Don't Tell Repeal Act of 2010. The American Recovery and Reinvestment Act of 2009 and Tax Relief, Unemployment Insurance Reauthorization, and Job Creation Act of 2010 served as economic stimulus amidst the Great Recession. After a lengthy debate over the national debt limit, he signed the Budget Control and the American Taxpayer Relief Acts. In foreign policy, he increased U.S. troop levels in Afghanistan, reduced nuclear weapons with the United States–Russia New START treaty, and ended military involvement in the Iraq War. He ordered military involvement in Libya, contributing to the overthrow of Muammar Gaddafi. He also ordered the military operations that resulted in the deaths of Osama bin Laden and suspected Yemeni Al-Qaeda operative Anwar al-Awlaki. After winning re-election by defeating Republican opponent Mitt Romney, Obama was sworn in for a second term in 2013. During this term, he promoted inclusiveness for LGBT Americans. His administration filed briefs that urged the Supreme Court to strike down same-sex marriage bans as unconstitutional (United States v. Windsor and Obergefell v. Hodges); same-sex marriage was fully legalized in 2015 after the Court ruled that a same-sex marriage ban was unconstitutional in Obergefell. He advocated for gun control in response to the Sandy Hook Elementary School shooting, indicating support for a ban on assault weapons, and issued wide-ranging executive actions concerning global warming and immigration. In foreign policy, he ordered military intervention in Iraq in response to gains made by ISIL after the 2011 withdrawal from Iraq, continued the process of ending U.S. combat operations in Afghanistan in 2016, promoted discussions that led to the 2015 Paris Agreement on global climate change, initiated sanctions against Russia following the invasion in Ukraine and again after Russian interference in the 2016 United States elections, brokered a nuclear deal with Iran, and normalized U.S. relations with Cuba. Obama nominated three justices to the Supreme Court: Sonia Sotomayor and Elena Kagan were confirmed as justices, while Merrick Garland faced unprecedented partisan obstruction and was ultimately not confirmed. During his term in office, America's soft power and reputation abroad significantly improved.Obama's presidency has generally been regarded favorably, and evaluations of his presidency among historians, political scientists, and the general public place him among the upper tier of American presidents. Obama left office and retired in January 2017 and currently resides in Washington, D.C. A December 2018 Gallup poll found Obama to be the most admired man in America for an unprecedented 11th consecutive year, although Dwight D. Eisenhower was selected most admired in twelve non-consecutive years."
                 }
             }
         }
     }'''
    #https://en.wikipedia.org/w/api.php?action=query&prop=description&titles=Barack_Obama
    
if __name__ == '__main__':
    inputText = "WASHINGTON (Reuters) - U.S. House Intelligence Committee Chairman Adam Schiff said on Monday he fully expected four White House officials scheduled for depositions with investigators in the House’s impeachment inquiry to defy congressional subpoenas. We expect the witnesses who have been subpoenaed to come in this afternoon, at White House instruction, also to be no-shows. This will only further add to the body of evidence on a potential obstruction of Congress charge against the president, Schiff told reporters. The officials were called in to testify in the House’s ongoing impeachment inquiry stemming from a July 25 call in which U.S. President Donald Trump pressed Ukrainian President Volodymyr Zelenskiy to investigate one of Trump’s domestic political rivals, former vice president and leading Democratic presidential candidate Joe Biden."
  
    LPDic = nerNgrammodule.loadLP()
     
    
    
    tokensList = nerNgrammodule.getTokensList(LPDic,inputText,5,0.01)
    
    
    #print(tokensList)
    print()
    #mcObj = MentionContext.MentionContext()
    
    Embedding = embeddingsmodule.loadEmbedding()
    
    CandEntities  = candentitiesmodule.loadCandidates(5)
    
    S = Similarity()
    ecObj = EntityContext.EntityContext()
    _entCtxtDic = ecObj.loadEntities()
 
    DisambiguationList = S.disambiguate(tokensList, inputText,CandEntities, Embedding, _entCtxtDic )
    print(DisambiguationList)
    
    #outputContent = S.outText(DisambiguationList,inputText)
    
    
 
 