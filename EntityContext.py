#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:55:27 2019

@author: renato
"""


import requests
import re
import gensim
import csv
#nltk.download('stopwords')
from nltk.corpus import stopwords
import urllib.parse

class EntityContext:
    
    #url_article = 'http://%s.wikipedia.org/w/index.php?action=raw&title=%s'
    #url_image = 'http://%s.wikipedia.org/w/index.php?title=Special:FilePath&file=%s'
    #url_search = 'http://%s.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&sroffset=%d&srlimit=%d&format=yaml'
    #url = 'http://en.wikipedia.org/w/api.php?action=parse&page=%s&format=json&prop=text&section=%s' % (topic, str(n))

    def getWikiSection(self, entity):
        cachedStopWords = stopwords.words('english')
        #entity = "|".join([str(urllib.parse.quote(e)) for e in elist])
        #pList = []
        entity = urllib.parse.quote(entity)
        try:
            print(entity)
            url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles=%s"%(entity)
            print(url)
            response = requests.get(url)
            if response.status_code == 200:
                json_response = response.json()
                query = json_response['query']
                page = query['pages']
                #for page in pages:
                firstKey = list(page.keys())[0]
                if firstKey != '-1':
                    jsonObj = page[firstKey]
                    text = jsonObj['extract']
                    text = self.unhtml(text)
                    text = self.unwiki(text)
                    text = self.punctuate(text)
                    text = text.split()
                    text = text[0:100]
                    text = ' '.join([word for word in text if word not in cachedStopWords])
                    #pList.append(text)
                    return text
                else:
 #                   continue
                    return None

            elif response.status_code == 404:
                return None
                #pass
        except:
            #pass
            return None
        #finally:
        #    return text
    
    def getContextWindow(self,sentence, cachedStopWords, win_size):
        '''

        Parameters
        ----------
        sentence : TYPE
            DESCRIPTION.
        cachedStopWords : TYPE
            DESCRIPTION.
        win_size : TYPE
            DESCRIPTION.

        Returns
        -------
        List 
            DESCRIPTION.

        '''
        
        sentence = self._remove_non_printed_chars(sentence)
        #removing stop words
        sentence = self._remove_stop_words(sentence,cachedStopWords)
        #striping out words shorter than min_len and longer than max_len 
        sentence = ' '.join(gensim.utils.simple_preprocess(sentence,min_len=4,max_len=10))    
        sentence = sentence.split()
        return sentence[:win_size]
    
        
    def _remove_non_printed_chars(self,string):
        reg = re.compile('[^a-zA-Zа-яА-ЯёЁ]')
        return reg.sub(' ', string)

    def _remove_stop_words(self,sent,cachedStopWords):       
        #cachedStopWords = stopwords.words('english')
        sent = ' '.join([word for word in sent.split() if word not in cachedStopWords])
        return sent

    def _trim_string(self, string):
       # remove extra spaces, remove trailing spaces, lower the case
       return re.sub('\s+',' ',string).strip().lower()


    def unhtml(self,html):
        """
        Remove HTML from the text.
        """
        html = re.sub(r'(?i)&nbsp;', ' ', html)
        html = re.sub(r'(?i)<br[ \\]*?>', '\n', html)
        html = re.sub(r'(?m)<!--.*?--\s*>', '', html)
        html = re.sub(r'(?i)<ref[^>]*>[^>]*<\/ ?ref>', '', html)
        html = re.sub(r'(?m)<.*?>', '', html)
        html = re.sub(r'(?i)&amp;', '&', html)
       
        return html
    
    
    def unwiki(self,wiki):
        """
        Remove wiki markup from the text.
        """
        wiki = re.sub(r'(?i)\{\{IPA(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), wiki)
        wiki = re.sub(r'(?i)\{\{Lang(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), wiki)
        wiki = re.sub(r'\{\{[^\{\}]+\}\}', '', wiki)
        wiki = re.sub(r'(?m)\{\{[^\{\}]+\}\}', '', wiki)
        wiki = re.sub(r'(?m)\{\|[^\{\}]*?\|\}', '', wiki)
        wiki = re.sub(r'(?i)\[\[Category:[^\[\]]*?\]\]', '', wiki)
        wiki = re.sub(r'(?i)\[\[Image:[^\[\]]*?\]\]', '', wiki)
        wiki = re.sub(r'(?i)\[\[File:[^\[\]]*?\]\]', '', wiki)
        wiki = re.sub(r'\[\[[^\[\]]*?\|([^\[\]]*?)\]\]', lambda m: m.group(1), wiki)
        wiki = re.sub(r'\[\[([^\[\]]+?)\]\]', lambda m: m.group(1), wiki)
        wiki = re.sub(r'\[\[([^\[\]]+?)\]\]', '', wiki)
        wiki = re.sub(r'(?i)File:[^\[\]]*?', '', wiki)
        wiki = re.sub(r'\[[^\[\]]*? ([^\[\]]*?)\]', lambda m: m.group(1), wiki)
        wiki = re.sub(r"''+", '', wiki)
        wiki = re.sub(r'(?m)^\*$', '', wiki)
       
        return wiki
       
    def punctuate(self,text):
        """
        Convert every text part into well-formed one-space
        separate paragraph.
        """
        text = re.sub(r'\r\n|\n|\r', '\n', text)
        text = re.sub(r'\n\n+', '\n\n', text)
       
        parts = text.split('\n\n')
        partsParsed = []
       
        for part in parts:
            part = part.strip()
           
            if len(part) == 0:
                continue
           
            partsParsed.append(part)
       
        return '\n\n'.join(partsParsed)


    def loadEntities(self):
        _eDic = {}
        fin = open("./entities.csv", "r", 1, encoding='utf-8')
        reader = csv.reader(fin,delimiter=',', quoting=csv.QUOTE_ALL)
        #mentionDic = {}
        for row in reader:
            _docid = row[0]
            _text = row[1]
            _eDic[_docid] = _text#print(len(_text))
        return _eDic

        
    def dumpEntities(self):       
        entSet = set()
        
        ecObj = EntityContext()

        
        for file in ["/home/renato/datasets/conll/conllYAGO_testb_GT_NONIL.tsv", "/home/renato/datasets/ace2004/ace2004_GT.tsv","/home/renato/datasets/iitb/iitb_GT_NONIL.tsv"]:
            f = open(file, 'r',1,encoding='utf-8')            
            for i in f:
                i = i.lower()
                doc, mention, offset, el = i.split('\t')
                entSet.add(el.lower().strip())
            f.close()
        f = open("/home/renato/datasets/msnbc/MSNBC_GT.tsv", 'r',1,encoding='windows-1252')            
        for i in f:
            i = i.lower()
            doc, mention, offset, el = i.split('\t')
            entSet.add(el.lower().strip())
        f.close()
        
        
        fout = open("./entities.csv", "w", 1, encoding='utf-8')
        fwriter = csv.writer(fout,delimiter=',', quoting=csv.QUOTE_ALL)
        for _item in entSet:
            extract = ecObj.getWikiSection(_item)
            if extract != None:
                if len(extract) > 100:
                    fwriter.writerow((_item,extract))
                else:
                    continue#pass
        fout.flush()
        fout.close()
        
if __name__ == "__main__":
    elist = ["vice president of brazil","premier of the soviet union","democratic national committee","white house","testify","donald trump"]
    #print(str("|".join([str(urllib.parse.quote(e)) for e in elist])))
    #for e in elist:
    EC = EntityContext()
    sent = EC.getWikiSection("vice president of brazil")
    
    #EC.dumpEntities()
    #_eDic = EC.loadEntities()
    
    #print(_eDic['vice president of brazil'])
    print(sent)
    print()
        
    #cachedStopWords = stopwords.words('english')
    #if sent != None:
    #    print(EC.getContextWindow(sent,cachedStopWords,100))
            

        