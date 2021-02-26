#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:25:27 2019

@author: renato
"""

import flask
from flask import Flask, render_template,request,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import nermodule
import candentitiesmodule
import embeddingsmodule
import EntityContext
import MethodPrior 
import MethodSimilarity
import MethodSimilarityPrior
import nerNgrammodule
import agreementmodule
from flask import g
import logging

app = Flask(__name__)

#class Todo(db.Model):
#    id = db.Column(db.Integer, primary_key=True)
#    content = db.Column(db.String(200), nullable=False)
#    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    

#    def __repr__(self):
#        return '<Task %r>' %self.id
    
# This runs before every request
@app.before_request
def before_request():
    g.user = "Renato is the user"
    

@app.route('/',methods=['POST','GET'])
def index(): 

#    if request.method == 'POST':
#        task_content = request.form['content']
#        new_task = Todo(content = task_content)
#        try:
#            db.session.add(new_task)
#            db.session.commit()
#            return redirect('/')
#        except:
#            return 'There was an issue adding your task'
#        
#    else:        
#        tasks = Todo.query.order_by(Todo.date_created).all()
#    
    return render_template('index.html')
    
    
@app.route('/submitTask',methods=['POST','GET'])
def submitTask():   
    #Pmodel = g.Pmodel
    if request.method == 'POST' :
        option = request.form['exampleFormControlSelect1']
        logging.warning(option)  # will print a message to the console

        if not option:
            #return render_template('result.html', inputText = content, selectedOption = option, content =  flask.Markup(outputContent))
            return render_template('index.html')
        option = int(option)
        content = request.form['inputText']
        if not content:
            #print(content) 
            return render_template('index.html')
        #return 'There was an issue deleting your task'
        else:
            
            if option == 1:
                outputContent = agreementmodule.agreementCalculation(content)
                if not outputContent:
                    return render_template('index.html')
                else:
                    return render_template('result.html', inputText = content, selectedOption = option, content =  flask.Markup(outputContent))
            
            if option == 2:
                tokList = nermodule.entityRecognise(content)
                outputContent =  nermodule.outText(tokList,content)
                if not outputContent:
                    return render_template('index.html')
                else:
                    return render_template('result.html', inputText = content, content =  flask.Markup(outputContent))
            
            if option == 3:
                
                #text,max_token,threshold
                tokensList =  nerNgrammodule.getTokensList(LPDic,content,2,0.05)
                logging.warning(tokensList)
                outputContent = nerNgrammodule.outText(tokensList,content)
                if not outputContent:
                    return render_template('index.html')
                else:


                    return render_template('result.html', inputText = content, selectedOption = option, content =  flask.Markup(outputContent))
            
            
            if option == 4:
                P = MethodPrior.Priormodel() 
                
                #tokensList = nermodule.getTokensList(content)
                tokensList =  nerNgrammodule.getTokensList(LPDic,content,3,0.05)
                DisambiguationList = P.disambiguate(tokensList,CandEntities)
                outputContent =  P.outText(DisambiguationList,content)
                return render_template('result.html', inputText = content, selectedOption = option, content =  flask.Markup(outputContent))

            
            if option == 5:
                S = MethodSimilarity.Similarity()
                #tokensList = nermodule.getTokensList(content)
                tokensList =  nerNgrammodule.getTokensList(LPDic,content,3,0.05)
                DisambiguationList = S.disambiguate(tokensList, content, CandEntities, Embedding)
                outputContent = S.outText(DisambiguationList,content)
                return render_template('result.html', inputText = content, selectedOption = option, content =  flask.Markup(outputContent))
            
            if option == 6:
                SP = MethodSimilarityPrior.SimilarityPrior()
                #tokensList = nermodule.getTokensList(content)
                tokensList =  nerNgrammodule.getTokensList(LPDic,content,3,0.05)
                DisambiguationList = SP.disambiguate(tokensList,content, CandEntities, Embedding)
                outputContent = SP.outText(DisambiguationList,content)
                return render_template('result.html', inputText = content, selectedOption = option, content =  flask.Markup(outputContent))
 
            if option == 7:
                return("Work in progress.")
                    
          
                
            

#@app.route('/delete/<int:id>')
#def delete(id):      
#    task_to_delete = Todo.query.get_or_404(id)    
#    try:
#            db.session.delete(task_to_delete)
#            db.session.commit()
#            return redirect('/')
#    except:
#            return 'There was an issue deleting your task'
        

#@app.route('/update/<int:id>',methods=['POST','GET'])
#def update(id):    
#    task = Todo.query.get_or_404(id)
#    if request.method == 'POST' :        
#       task.content = request.form['content']                
#       try:
#           db.session.commit()
#           return redirect('/')
#       except:
#           return 'There was an issue updating your task'
#   else :
#       return render_template('update.html',task=task)

#with app.app_context():
#    init_db()


if __name__ == "__main__":
    
    global Embedding
    Embedding = embeddingsmodule.loadEmbedding()
    logging.warning('Embeddings Loaded')  # will print a message to the console

    
    global CandEntities
    CandEntities  = candentitiesmodule.loadCandidates(20)
    logging.warning('Candidates Loaded')  # will print a message to the console
    
    global LPDic
    LPDic = nerNgrammodule.loadLP()
    logging.warning('Link Probability Loaded')  # will print a message to the console
                    

    
    
    #global Pmodel
    #Pmodel = MethodPrior.Priormodel().load_model()
    #logging.warning('Prior Loaded!')  # will print a message to the console

    #logging.warning('Loading Candidates...')  # will print a message to the console
    #global CandEntities
    #CandEntities  = candentitiesmodule.loadCandidates(3)
    #logging.warning('Candidates Loaded!')  # will print a message to the console
    
    #global Embedding
    #Embedding = embeddingsmodule.loadEmbedding()
    #logging.warning('Embeddings Loaded')  # will print a message to the console

    
    app.run(host='0.0.0.0', port=8484, debug=True)

    #app.create()

