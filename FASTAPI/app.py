from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np
import pandas as pd


app = FastAPI()
model_name = 'Classification_Model.joblib'
model = joblib.load(model_name)

class Loan(BaseModel):
    Gender: float
    Married: float
    Dependents: float
    Education: float
    Self_Employed: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: float
    TotalIncome: float  

@app.get('/')
def index():
    return {'welcome to Loan Prediction'}  

@app.post('/prediction')
def predict(laon_details: Loan):
    data = laon_details.model_dump()
    gender = data['Gender']
    married = data['Married']
    dependents = data['Dependents']
    education = data['Education']
    self_employed = data['Self_Employed']
    loan_amt = data['LoanAmount']
    loan_term = data['Loan_Amount_Term']
    credit_hist = data['Credit_History']
    property_area = data['Property_Area']
    income = data['TotalIncome']


    prediction = model.predict([[gender,married,dependents,education,self_employed,
                   loan_amt,loan_term,credit_hist,property_area,income]])

    if prediction == 1:
        pred = 'Approved'
    else:
        pred = 'Rejected'

    return {"STATUSE OF YOUR LOAN APPLICATION IS":pred}



if __name__== '__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)
    