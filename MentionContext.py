#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:55:17 2019

@author: renato
"""
#nltk.download('stopwords')
from nltk.corpus import stopwords

class MentionContext:
    
    cachedStopWords = stopwords.words('english')

    def getContextWindow(self, sentence, phrase, pos, win_size):
        '''
        Parameters
        ----------
        sentence : String
            The input raw text.
        phrase : String
            mention, token, unigram, bigram, etc
        pos : TYPE
            DESCRIPTION.
        win_size : int
            Size of the desired window of words surrounding the mention of interest.

        Returns
        -------
        The list of tokes within the windows delimitation.

        '''
        
        # if the word is in the middle
           # there is not enough words ahead 
           # there is not enough words behind
        
        # if the word is the last word
        # if the word is the first word
        # if the word is not present 
        phraseLen = len(phrase)
        totalSize = len(sentence)
        #print(sentence.find(phrase)) #  10 and 45
        #pos = int(sentence.find(phrase))
        leftW = sentence[0:pos]
        #print(leftW)
        rightW = sentence[pos+phraseLen:totalSize]
        #print(rightW)    
        
    
        leftW = ' '.join([word for word in leftW.split() if word not in self.cachedStopWords])
        leftW  = leftW .split()
        
        rightW = ' '.join([word for word in rightW.split() if word not in self.cachedStopWords])
        rightW = rightW.split()
        
        #print()
        #print(leftW)
        #print(rightW)
        #print(len(leftW))
        #win_size = 2
        
        if len(leftW) < win_size:
            leftW  = leftW[:]
        else:
            leftW  = leftW[len(leftW)-win_size:]
       
        if len(rightW) < win_size:
            rightW = rightW[:]
        else:
            rightW = rightW[:win_size]
        
        final = []
        final.append(' '.join(str(i) for i in leftW ))
        final.append(phrase)
        final.append(' '.join(str(i) for i in rightW ))
    #print()
    #print(leftW)
    #print(rightW)    
    #print()
    #print(final)    
        final = ' '.join(str(i) for i in final )   
        final = final.split()
        return final
    
 
if __name__ == "__main__":
    sentence = "Iran will protest to the International Court protest of Justice at the Hague and the in New York."
    phrase = "International Court"
    phraseLen = len(phrase)
    totalSize = len(sentence)
    #print(sentence.find(phrase)) #  10 and 45
    #pos = int(sentence.find(phrase))
    pos = 45
    win_size = 5
    print(MentionContext.getContextWindow(sentence,phrase,pos,win_size))
    
    #cachedStopWords = stopwords.words('english')
    #sentence = ' '.join([word for word in sentence.split() if word not in cachedStopWords])
    #print(sentence)
    