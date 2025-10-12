import numpy as np
import pickle
import pandas as pd
from flask import Flask,request,render_template

app=Flask(__name__, static_url_path='/Flask/static')
model= pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=["POST"])
def predict():
    Gender=float(request.form["Gender"])
    Hemoglobin=float(request.form["Hemoglobin"])
    MCH=float(request.form['MCH'])
    MCHC=float(request.form['MCHC'])
    MCV=float(request.form['MCV'])
    features_values=np.array([[Gender,Hemoglobin,MCH,MCHC,MCV]])
    df=pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin','MCH','MCHC','MCV'])
    print(df)
    prediction=model.predict(df)
    print(prediction[0])
    result=prediction[0]
    if prediction[0]==0:
        result="You don'y have any anemia disease"
    elif prediction[0]==1:
        result="You have anemia disease"
    text="Hence, based on calculation: "
    return render_template("predict.html",prediction_text=text+str(result))
if __name__ == "__main__":
    app.run(debug=False, port=5000)