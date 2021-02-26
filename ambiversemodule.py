#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:59:01 2019

@author: renato
"""

# importing the requests library 
import requests 
import json
import pandas


def annotate(inputText):
    service_url = "https://api.ambiverse.com/v2/entitylinking/analyze"
    conn_timeout = 1
    read_timeout = 5
    max_retries = 5
    timeouts = (conn_timeout, read_timeout)

    body = {'docId': 'false',
            'text' : inputText,
            'language' : 'en',
            'extractConcepts': 'true'
            }

    headers = {'content-type': 'application/json', 
               'accept':'application/json',
               'authorization': '#REQUIRED_TOKEN'
               }
    dfObj = pandas.DataFrame(columns=['mention','position','eid'])
    for _ in range(max_retries):
        try:
            r = requests.post(url = service_url,  headers = headers,  data=json.dumps(body), timeout = timeouts)  
            pastebin_url = r.text 
            data = json.loads(pastebin_url)
            matchesFragment = data.get('matches')
            entitiesFragment = data.get('entities')
            entDic = {}
            for match in matchesFragment:
                mention = match.get('text')
                eid = match.get('entity').get('id')
                pos = match.get('charOffset')
                if eid :
                    entDic[eid] = match.get('text') +'\t' + str(match.get('charOffset'))
                    dfObj = dfObj.append({'mention':mention, 'position': pos, 'eid': eid}, ignore_index=True)
                eDic = {}#print()
                for ent in entitiesFragment:
                    try:
                        eLink = ent.get('name') 
                        eid = ent.get('id') 
                        eDic[eid] = eLink
                    except:
                        print("Empty element")
                        pass
                for key, value in eDic.items():
                    dfObj.loc[(dfObj.eid == key),'eid'] = value
            break
        except:
            print("pass {}".format(_))
            pass
    dfObj = dfObj.sort_values(by=['position'])
    return dfObj      

if __name__ == "__main__":
    #inputText = "Wall Street Sounds the Alarm With a populist message that promises to rein in corporate excess, Ms. Warren has been facing more hostility from the finance industry than any other candidate."
    inputText = "KABUL, Afghanistan (AP) _ The ruling Taliban militia on Monday released 137 Shiite Muslim prisoners it had held for nearly two years and urged the opposition to follow suit and release government prisoners it is holding. The freed men, all said to be fighters belonging to the opposition alliance, were released ahead of the Islamic holy month of Ramadan, when devout Muslims fast from sunrise to sunset. ``The prisoners are being released as a gesture of kindness'' by the Taliban's supreme leader Mullah Mohammed Omar, said the Taliban's Interior Minister Abdul Razzak Akhund. ``We ask the opposition to show their heart and release government prisoners in their jails.'' The opposition alliance, which controls barely five percent of Afghanistan and is fighting a war against the dominant Taliban, is mostly made up of the country's minority ethnic and religious groups. The Taliban are predominantly Sunni Muslim, which is the majority Islamic sect in the country. The Taliban have been accused by international human rights groups of mistreating the minority Shiite Muslims, a charge they reject. The newly released prisoners were captured when the Taliban took control of Afghanistan's central Bamyan province, nearly two years ago. The area is largely inhabited by minority Shiite Muslims. ``I am very happy to be going home,'' said Ali Jan, one of the prisoners who said he was a farmer in Bamyan and not a soldier."
    AnnotationsDic = annotate(inputText)
    print()
    print(AnnotationsDic)