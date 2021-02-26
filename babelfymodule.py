#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:26:41 2019

@author: renato
"""
import urllib
import urllib.request
from urllib.parse import urlencode
import json
import gzip
import pandas as pd
from io import BytesIO
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def annotate(inputText):
    service_url = 'https://babelfy.io/v1/disambiguate'
    #inputText = 'BabelNet is both a multilingual encyclopedic dictionary and a semantic network'
    lang = 'EN'
    key  = '#REQUIRED_KEY'

    max_retries = 5
    params = {
        'text' : inputText,
        'lang' : lang,
        'key'  : key
        }

    url = service_url + '?' + urlencode(params)
    for _ in range(max_retries):
        try:
            #request = urllib2.Request(url)
            request = urllib.request.Request(url)
            #request = urllib2.Request(url)
            request.add_header('Accept-encoding', 'gzip')
            #response = urllib2.urlopen(request)
            response = urllib.request.urlopen(request)
            dfObj = pd.DataFrame(columns=['mention','position','eid'])
            if response.info().get('Content-Encoding') == 'gzip':
                buf = BytesIO(response.read())
                f = gzip.GzipFile(fileobj=buf)
                data = json.loads(f.read())
            	    # retrieving data
                for result in data:
                    # retrieving token fragment
                    #tokenFragment = result.get('tokenFragment')
                    #tfStart = tokenFragment.get('start')
                    #tfEnd = tokenFragment.get('end')
                    # retrieving char fragment
                    charFragment = result.get('charFragment')
                    cfStart = charFragment.get('start')
                    cfEnd = charFragment.get('end')
                    #retrieving BabelSynset ID
                    #synsetId = result.get('babelSynsetID')
                    #retrieving DBpediaURL
                    DBpediaURL = result.get('DBpediaURL')
                    if DBpediaURL:
                        #URI_BASE = u"http://dbpedia.org/resource/{}"
                        entity_title = DBpediaURL.replace("http://dbpedia.org/resource/", "").replace("_", " ")
                        mention = inputText[cfStart:cfEnd+1]
                        pos = cfStart
                        #AnnotationsDic[mention+'\t'+str(pos)] = entity_title
                        dfObj = dfObj.append({'mention':mention, 'position': pos, 'eid': entity_title}, ignore_index=True)
            break
        except Exception as e:
            print("pass {}".format(_))
            print(e)
            pass
    
    dfObj = dfObj.sort_values(by=['position'])            
    return dfObj    

if __name__ == "__main__":
    #inputText = "how"
    inputText = "KABUL, Afghanistan (AP) _ The ruling Taliban militia on Monday released 137 Shiite Muslim prisoners it had held for nearly two years and urged the opposition to follow suit and release government prisoners it is holding. The freed men, all said to be fighters belonging to the opposition alliance, were released ahead of the Islamic holy month of Ramadan, when devout Muslims fast from sunrise to sunset. ``The prisoners are being released as a gesture of kindness'' by the Taliban's supreme leader Mullah Mohammed Omar, said the Taliban's Interior Minister Abdul Razzak Akhund. ``We ask the opposition to show their heart and release government prisoners in their jails.'' The opposition alliance, which controls barely five percent of Afghanistan and is fighting a war against the dominant Taliban, is mostly made up of the country's minority ethnic and religious groups. The Taliban are predominantly Sunni Muslim, which is the majority Islamic sect in the country. The Taliban have been accused by international human rights groups of mistreating the minority Shiite Muslims, a charge they reject. The newly released prisoners were captured when the Taliban took control of Afghanistan's central Bamyan province, nearly two years ago. The area is largely inhabited by minority Shiite Muslims. ``I am very happy to be going home,'' said Ali Jan, one of the prisoners who said he was a farmer in Bamyan and not a soldier."
    dfObj = annotate(inputText)
    print(dfObj)
    

    