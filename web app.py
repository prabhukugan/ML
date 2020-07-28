
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
import json

filename = 'Log_Reg_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

app = Flask(__name__) # initializing a flask app

@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':

            Pregnancies = int(request.form['Pregnancies'])
            Glucose = int(request.form['Glucose'])
            BloodPressure = int(request.form['BloodPressure'])
            SkinThickness = int(request.form['SkinThickness'])
            Insulin = int(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age=int(request.form['Age'])

            data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age]])
            prediction = loaded_model.predict(data)

            return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app