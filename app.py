import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model1 = pickle.load(open('mainprojectNaiveBayes.pkl','rb'))
model2 = pickle.load(open('mainprojectlog.pkl','rb'))
model3 = pickle.load(open('mainproject.pkl','rb'))
model4 = pickle.load(open('mainprojectforest.pkl','rb'))

@app.route('/')
def home():
  
    return render_template("index.html")

@app.route('/aboutme')
def aboutme():
    return render_template('aboutme.html')
  
@app.route('/predict',methods=['GET'])
def predict():
    numeric1 = float(request.args.get('numeric1'))
    numeric2 = float(request.args.get('numeric2'))
    numeric3 = float(request.args.get('numeric3'))
    numeric4 = float(request.args.get('numeric4'))
    numeric5= float(request.args.get('numeric5'))
    
    Model = (request.args.get('Model'))
    if Model=="Naive Bayes Algorithm":
      input_pred = model1.predict([[numeric1,numeric2,numeric3,numeric4,numeric5]])
      input_pred = input_pred.astype(int)
      print(input_pred)

    elif Model=="Logistic Regression Algorithm":
      input_pred = model2.predict([[numeric1,numeric2,numeric3,numeric4,numeric5]])
      input_pred = input_pred.astype(int)
      print(input_pred)

    elif Model=="Decision Tree Algorithm":
      input_pred = model3.predict([[numeric1,numeric2,numeric3,numeric4,numeric5]])
      input_pred = input_pred.astype(int)
      print(input_pred)
      
    else:
      input_pred = model4.predict([[numeric1,numeric2,numeric3,numeric4,numeric5]])
      input_pred = input_pred.astype(int)
      print(input_pred)

    if input_pred[0]==1:
        result= "Legitimite"
    else:
        result="Phising" 

    return render_template('index.html', prediction_text='Phising Anlaysis: {}'.format(result))



if __name__=="__main__":
    app.run(debug=True)
