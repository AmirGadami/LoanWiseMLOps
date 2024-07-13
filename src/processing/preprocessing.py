from sklearn.base import BaseEstimator, TransformerMixin
import pathlib
from pathlib import Path
import os
from configs import config
import sys
import numpy as np


sys.path.append(str(config.PATH_ROOT))




class MEAN_IMPUTER(BaseEstimator,TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X):
        self.mean_col = {}
        for col in self.variables:
            self.mean_col[col] = X[col].mean()
        return self 
    
    def transform(self,X):
        for col in self.variables:
            X = X.copy()
            X[col].fillna(self.mean_col[col],inplace=True)
        return X


class MODE_IMPUTER(BaseEstimator,TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X):
        self.mode_col = {}
        for col in self.variables:
            self.mode_col[col] = X[col].mode()
        return self 
    
    def transform(self,X):
        for col in self.variables:
            X = X.copy()
            X[col].fillna(self.mode_col[col],inplace=True)
        return X
    

class DropColumns(BaseEstimator,TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self,X):
        return self
    
    def transform(self,X):
        X = X.copy()
        X.drop(columns = self.variables)
        return X


class COLUMN_DROPPER(BaseEstimator,TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X):
        self.mode_col = {}
        for col in self.variables:
            self.mode_col[col] = X[col].mode()
        return self 
    
    def transform(self,X):
        for col in self.variables:
            X = X.copy()
            X[col].fillna(self.mode_col[col],inplace=True)
        return X
    
class DomainProcessing(BaseEstimator,TransformerMixin):
    def __init__(self,variables):
        self.variables = variables

    def fit(self,X):
        self.label_dict = {}
        for col in self.variables:
            t = X[col].value_counts().sort_values(ascending=True).index 
            self.label_dict[col] = {v:k for v,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].map(self.label_dict[col]) 
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
            X[feature] = X[feature] + X[self.variable_to_add]
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
            


