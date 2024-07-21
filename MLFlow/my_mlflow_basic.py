import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numbers as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def load_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url,sep=";")
    return df

def eval(actual,pred):
    mae = mean_absolute_error(actual,pred)
    rmse = mean_squared_error(actual,pred,squared=False)
    r2 = r2_score(actual,pred)
    return mae,rmse,r2

def main(alpha,l1_ratio):
    data = load_data()
    TARGET = 'quality'
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

    mlflow.set_experiment("MYMLFLOW-BASIC")
    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        mae,rmse,r2 = eval(y_test,y_pred)

        mlflow.log_param("alpha",alpha)
        mlflow.log_param('l1_ratio',l1_ratio)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("r2",r2)
        mlflow.sklearn.log_model(model,"model")




        




if __name__ == '__main__':
    data = load_data()
    print(data)
    args = argparse.ArgumentParser()
    args.add_argument('--alpha','-a',type=float, default=0.2)
    args.add_argument('--l1_ratio','-l1', type=float,default=0.3)
    parsed_args = args.parse_args()
    main(parsed_args.alpha, parsed_args.l1_ratio)