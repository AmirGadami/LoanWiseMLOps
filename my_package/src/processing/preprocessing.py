from sklearn.base import BaseEstimator, TransformerMixin
import pathlib
from pathlib import Path
import os
from my_package.configs import config
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")


sys.path.append(str(config.PATH_ROOT))



class TYPE_FIX(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X,y=None):
        
        return self


    def transform(self,X):
        X = X.copy()
        for col in self.variables:
                
            if X[col].dtypes == 'int64' :
                X[col] = X[col].astype('float64')
        return X


class MEAN_IMPUTER(BaseEstimator,TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X,y=None):
        self.mean_col = {}
        for col in self.variables:
            self.mean_col[col] = X[col].mean()
        return self 
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            
            X[col].fillna(self.mean_col[col],inplace=True)
        return X


class MODE_IMPUTER(BaseEstimator,TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X,y=None):
        self.mode_col = {}
        for col in self.variables:
            self.mode_col[col] = X[col].mode()[0]
        return self 
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            
            X[col].fillna(self.mode_col[col],inplace=True)
        return X
    

class COLUMN_DROPPER(BaseEstimator,TransformerMixin):

    def __init__(self,variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X = X.drop(columns = self.variables_to_drop)
        return X

    
class CustomLabelEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,variables):
        self.variables = variables

    def fit(self, X,y=None):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.label_dict[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    
    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X
    
class DomainProcessing(BaseEstimator,TransformerMixin):

    def __init__(self,variable_to_modify = None, variable_to_add = None):
        self.variable_to_modify = variable_to_modify
        self.variable_to_add = variable_to_add
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for feature in self.variable_to_modify:
            X[feature] = X[feature].values + X[self.variable_to_add].values
        return X
    
class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = np.log(X[col])
        return X
            


if __name__ == "__main__":

    from src.processing.data_handling import load_data,save_pipeline
    import numpy as np
    import pandas as pd
    import sys


    train = load_data(config.TRAIN_FILE)
    train_x = train[config.FEATURES]
    train_y = train[config.TARGET].map({'N':0,'Y':1})

    test = load_data(config.TEST_FILE)
    test_x = test[config.FEATURES]
    # print(test.head())
    # test_y = test[config.TARGET].map({'N':0,'Y':1})  

    print(type(train))
    tf = TYPE_FIX(config.FEATURES)
    x = tf.transform(train)
 


    me = MEAN_IMPUTER(config.NUM_FEATURES)
    me.fit(x)
    x= me.transform(x)
    x_test_1= me.transform(test_x)
    # print(x_test_1.isna().sum())

    mo = MODE_IMPUTER(config.CAT_FEATURES)
    mo.fit(x)
    x= mo.transform(x)
    x_test_2= mo.transform(x_test_1)
    # print(x_test_2.isna().sum())

    ce = CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)
    ce.fit(x)
    x = ce.transform(x)
    x_test_3= ce.transform(x_test_2)
    # print(x_test_3.isna().sum())

    dp = DomainProcessing(variable_to_modify = config.FEATURES_TO_MODIFY, variable_to_add = config.FEATURES_TO_ADD)
    dp.fit(x)
    x= dp.transform(x)
    x_test_4= dp.transform(x_test_3)
    # print(x_test_4.isna().sum())



    lt = LogTransforms(variables=config.FEATURES_TO_LOG)
    lt.fit(x)
    x = lt.transform(x)
    x_test_6= lt.transform(x_test_4)
    # print(x_test_6.isna().sum())





