#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:13:04 2019

@author: renato
"""

import requests 
import json
import pandas

def annotate(inputText):
    # Set the authorization token for subsequent calls.
    service_url = "https://tagme.d4science.org/tagme/tag"

    conn_timeout = 1
    read_timeout = 5
    max_retries = 5

    timeouts = (conn_timeout, read_timeout)

    body = {'text' : inputText, 'language' : 'en'}

    headers = {'gcube-token' : '3784c0fd-fdd2-4cf4-83b8-269de5d7b49e-843339462'}

    dfObj = pandas.DataFrame(columns=['mention','position','eid'])
    for _ in range(max_retries):
        try:			         
            # sending post request and saving response as response object 
            r = requests.post(url = service_url,  headers = headers,  data=body, timeout = timeouts)  
            # extracting response text  
            data = r.text 
            data = json.loads(data)
		    #AnnotationsDic = {}

            annotationsFragment = data.get('annotations')
    		
            for ann_json in annotationsFragment:
                #entity_id = int(ann_json.get("id"))
                #score = float(ann_json.get("rho"))
                mention = ann_json.get("spot")
                pos =  int(ann_json.get("start"))
                entity_title = ann_json.get("title")
                #print("%s %d %s" %(mention, pos, entity_title))
                #AnnotationsDic[mention+'\t'+str(pos)] = entity_title
                dfObj = dfObj.append({'mention':mention, 'position': pos, 'eid': entity_title}, ignore_index=True)
            break
        except:
            print("pass {}".format(_))
            pass
    dfObj = dfObj.sort_values(by=['position'])
    return dfObj


if __name__ == "__main__":
    inputText = "WASHINGTON (Reuters) - U.S. House Intelligence Committee Chairman Adam Schiff said on Monday he fully expected four White House officials scheduled for depositions with investigators in the House’s impeachment inquiry to defy congressional subpoenas. We expect the witnesses who have been subpoenaed to come in this afternoon, at White House instruction, also to be no-shows. This will only further add to the body of evidence on a potential obstruction of Congress charge against the president, Schiff told reporters. The officials were called in to testify in the House’s ongoing impeachment inquiry stemming from a July 25 call in which U.S. President Donald Trump pressed Ukrainian President Volodymyr Zelenskiy to investigate one of Trump’s domestic political rivals, former vice president and leading Democratic presidential candidate Joe Biden."
    dfObj = annotate(inputText)
    print(dfObj)
    
    