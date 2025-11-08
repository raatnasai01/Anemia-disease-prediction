import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('Home.html')

@app.route('/predict')
def predict_page():
    return render_template('Predict.html')

@app.route('/result', methods=['POST'])
def predict():
    try:
        Gender = float(request.form['Gender'])
        Hemoglobin = float(request.form['Hemoglobin'])
        MCH = float(request.form['MCH'])
        MCHC = float(request.form['MCHC'])
        MCV = float(request.form['MCV'])
    except Exception as e:
        return render_template('result.html', result_text=f"Invalid input: {e}")

    features_values = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
    df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])

    prediction = model.predict(df)[0]
    if prediction == 0:
        result = "You don't have any Anemic disease"
    else:
        result = "You have anemic disease"

    return render_template('result.html', result_text=f"Hence, based on calculation: {result}")

if __name__ == "__main__":
    app.run(debug=True)
