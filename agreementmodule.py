#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:03:12 2019

@author: renato
"""


import ambiversemodule
import babelfymodule
import tagmemodule
import pandas


def agreementCalculation(inputText):
    
    
    BabelfyAnnotationsDF = babelfymodule.annotate(inputText)
    TagmeAnnotationsDF = tagmemodule.annotate(inputText)
    AmbiverseAnnotationsDF = ambiversemodule.annotate(inputText)
    

    if AmbiverseAnnotationsDF.empty == True or  BabelfyAnnotationsDF.empty == True or TagmeAnnotationsDF.empty== True :
        print('DataFrame is empty')
        return None
    else:
        print('DataFrame is not empty')
    
    easyMAP = {}
    mediumMAP = {}
    hardMAP = {}


    AmbiverseAnnotationsDic = {}
    BabelfyAnnotationsDic = {}
    TagmeAnnotationsDic = {}

    #print(AmbiverseAnnotationsDF)   
    print()
    #print(BabelfyAnnotationsDF)
    print()
    #print(TagmeAnnotationsDF)
    
    for index, row in AmbiverseAnnotationsDF.iterrows():
        tok = row['mention'] 
        pos = row['position']
        elink = row['eid']
        AmbiverseAnnotationsDic[tok+'\t'+str(pos)] = elink
    for index, row in BabelfyAnnotationsDF.iterrows():
        tok = row['mention'] 
        pos = row['position']
        elink = row['eid']
        BabelfyAnnotationsDic[tok+'\t'+str(pos)] = elink
    for index, row in TagmeAnnotationsDF.iterrows():
        tok = row['mention'] 
        pos = row['position']
        elink = row['eid']
        TagmeAnnotationsDic[tok+'\t'+str(pos)] = elink

    diffLIST = []
    

    
    print()
    print()
    print()
    #for item in AmbiverseAnnotationsDic.items():.
    
    dfObj = pandas.DataFrame(columns=['mention','position','diff'])
    
 
    
    for key, value in AmbiverseAnnotationsDic.items():
        if key in BabelfyAnnotationsDic:
            if key in TagmeAnnotationsDic:
                Alink = AmbiverseAnnotationsDic[key]
                Blink = BabelfyAnnotationsDic[key]
                Tlink = TagmeAnnotationsDic[key]
                
                #print(key)
                mention, pos = key.split('\t')
                if Alink.lower() != Blink.lower() and  Alink.lower() != Tlink.lower()  and  Blink.lower() != Tlink.lower() :
                    #print(key + "\t"+ Alink)
                    hardMAP[key] = Alink
                    
                    diffLIST.append(key+'\t'+ "HARD")
                    dfObj = dfObj.append({'mention':mention, 'position': pos, 'diff': 'HARD'}, ignore_index=True)
                    continue  
                    #outputText = substring.replace(tok,"<a href=\"#\" data-toggle=\"tooltip\" title=\""+typ+"\">"+tok+"</a>",1)
                    #outputText = substring.replace(mention,"<span class=\"label label-danger\" data-toggle=\"tooltip\" title=\"HARD\">"+mention+"</span>");


                if Alink.lower() == Blink.lower() and  Alink.lower() == Tlink.lower()  and  Blink.lower() == Tlink.lower() :
                    #print(key + "\t"+ Alink)
                    easyMAP[key] = Alink
                    diffLIST.append(key+'\t'+ "EASY")
                    dfObj = dfObj.append({'mention':mention, 'position': pos, 'diff': 'EASY'}, ignore_index=True)
                    continue
                    #outputText = substring.replace(mention,"<span class=\"label label-success\" data-toggle=\"tooltip\" title=\"EASY\">"+mention+"</span>");


                else:
                    #print(key + "\t"+ Alink)
                    mediumMAP[key] = Alink
                    diffLIST.append(key+'\t'+ "MEDIUM")
                    dfObj = dfObj.append({'mention':mention, 'position': pos, 'diff': 'MEDIUM'}, ignore_index=True)
                    #outputText = substring.replace(mention,"<span class=\"label label-info\" data-toggle=\"tooltip\" title=\"MEDIUM\">"+mention+"</span>");
                
                #previous = pos
                #@print(outputText)
                #finalText.append(outputText)
    
    
    print(diffLIST)
    finalText = []
    previous = len(inputText) 
    listSize = len(diffLIST)
    cont = 0
    #print(diffLIST)
    #print(dfObj)
    
  #  print()
  #  dfObj = dfObj.sort_values(by ='position' , ascending=True)
  #  print(dfObj)
 
    for elem in reversed(diffLIST):
        tok, pos, diff = elem.split('\t')
        pos = int(pos)
        cont+=1
        if cont == listSize:
            pos = 0
        substring = inputText[pos:previous]
        if diff.lower() == "easy":
            outputText = substring.replace(tok,"<span class=\"label label-success\" data-toggle=\"tooltip\" title=\""+diff+"\">"+tok+"</span>");
        if diff.lower() == "medium":
            outputText = substring.replace(tok,"<span class=\"label label-info\" data-toggle=\"tooltip\" title=\""+diff+"\">"+tok+"</span>");
        if diff.lower() == "hard":
            outputText = substring.replace(tok,"<span class=\"label label-danger\" data-toggle=\"tooltip\" title=\""+diff+"\">"+tok+"</span>");
        previous = pos
    #print(outputText)
        finalText.append(outputText)

    text = [str(lint) for lint in reversed(finalText)]
    text = "".join(text)
    return text
    


if __name__ == "__main__":
    inputText = "Brian Mawhinney, former Tory cabinet minister, dies aged 79 Mawhinney served as Conservative chairman from 1995 to 1997 under John Major"
    #inputText = "WASHINGTON (Reuters) - U.S. House Intelligence Committee Chairman Adam Schiff said on Monday he fully expected four White House officials scheduled for depositions with investigators in the House’s impeachment inquiry to defy congressional subpoenas. We expect the witnesses who have been subpoenaed to come in this afternoon, at White House instruction, also to be no-shows. This will only further add to the body of evidence on a potential obstruction of Congress charge against the president, Schiff told reporters. The officials were called in to testify in the House’s ongoing impeachment inquiry stemming from a July 25 call in which U.S. President Donald Trump pressed Ukrainian President Volodymyr Zelenskiy to investigate one of Trump’s domestic political rivals, former vice president and leading Democratic presidential candidate Joe Biden."
    #inputText = "As Warren Gains in Race, Wall Street Sounds the Alarm With a populist message that promises to rein in corporate excess, Ms. Warren has been facing more hostility from the finance industry than any other candidate "
    out_text = agreementCalculation(inputText)
    print()
    print(out_text)
    
    
    
   