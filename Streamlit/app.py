import streamlit as st
import pandas as pd
import numpy as np
import joblib
import wget

model_name = 'Classification_model.joblib'
wget.download('https://github.com/AmirGadami/LoanWiseMLOps/blob/main/Streamlit/Classification_Model.joblib')
model = joblib.load(model_name)

def prediction(Gender,Married,Dependents,
            Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
            LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
    if Gender == "Male":
        Gender = 1
    else:
        Gender = 0

    if Married == "Yes":
        Married = 1
    else:
        Married = 0

    if Education == "Graduate":
        Education = 0
    else:
        Education = 1

    if Self_Employed == "Yes":
        Self_Employed = 1
    else:
        Self_Employed = 0

    if Credit_History == "Outstanding Loan":
        Credit_History = 1
    else:
        Credit_History = 0   

    if Property_Area == "Rural":
        Property_Area = 0
    elif Property_Area == "Semi Urban":
        Property_Area = 1  
    else:
        Property_Area = 2  
    Total_Income = ApplicantIncome + CoapplicantIncome +1e-6
    Total_Income = np.log(Total_Income)

    p = model.predict([[Gender, Married, Dependents, Education, 
                        Self_Employed,LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,Total_Income]])
    print(p[0])

    return p

    # if p[0]==0:
    #     pred = "Rejected"

    # else:
    #     pred = "Approved"
    # return pred        


def main():
    st.title("Amir's First Web Model")
    st.header("Please enter your details to proceed with your loan Application")
    Gender = st.selectbox("Gender",("Male","Female"))
    Married = st.selectbox("Married",("Yes","No"))
    Dependents = st.number_input("Number of Dependents")
    Education = st.selectbox("Education",("Graduate","Not Graduate"))
    Self_Employed = st.selectbox("Self Employed",("Yes","No"))
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("LoanAmount")
    Loan_Amount_Term = st.number_input("Loan Amount Term")
    Credit_History = st.selectbox("Credit History",("Outstanding Loan", "No Outstanding Loan"))
    Property_Area = st.selectbox("Property Area",("Rural","Urban","Semi Urban"))

    if st.button('Predict'):
        predict = prediction(Gender,Married,Dependents,
            Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
            LoanAmount,Loan_Amount_Term,Credit_History,Property_Area) 

        if predict[0] == 1:
            st.success('Your loan Applicatioj is APPROVED!')
        else:
            st.error("Your Loan Aplication is REJECTED!")



if __name__ == "__main__":
    main()