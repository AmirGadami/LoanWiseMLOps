import pathlib
import os

PATH_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATAPATH = os.path.join(PATH_ROOT,'dataset')
SOURCE_PATH = os.path.join(PATH_ROOT,'src')
SAVE_MODEL_PATH = os.path.join(SOURCE_PATH,'models')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
MODEL_NAME = 'classification.pkl'
TARGET = 'Loan_Status'

FEATURES = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
            'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']
NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
CAT_FEATURES = ['Gender','Married','Dependents','Education',
                'Self_Employed','Credit_History','Property_Area']
FEATURES_TO_ENCODE = ['Gender','Married','Dependents','Education',
                'Self_Employed','Credit_History','Property_Area']
FEATURES_TO_DROP = ['CoapplicantIncome']
FEATURES_TO_ADD = ['CoapplicantIncome']
FEATURES_TO_LOG = ['ApplicantIncome', 'LoanAmount']
FEATURES_TO_MODIFY = ['ApplicantIncome']