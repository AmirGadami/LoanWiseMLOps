from my_package.configs import config
from sklearn.pipeline import Pipeline
import my_package.src.processing.preprocessing as pp 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression




classification_pipeline = Pipeline(
    
    [   
        ("TYPE FIX" , pp.TYPE_FIX(variables=config.FEATURES)),
        ("Domain Processing", pp.DomainProcessing(variable_to_modify=config.FEATURES_TO_MODIFY,
                                                  variable_to_add=config.FEATURES_TO_ADD)),
        ("Drop Column" , pp.COLUMN_DROPPER(variables_to_drop=config.FEATURES_TO_DROP)),
        ("MEAN IMPUTER", pp.MEAN_IMPUTER(config.NUM_FEATURES)),
        ("MODE IMPUTER", pp.MODE_IMPUTER(config.CAT_FEATURES)),
        ("Encode Labels" ,pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ("LOG", pp.LogTransforms(variables=config.FEATURES_TO_LOG)),
        ("SCALER", MinMaxScaler()),
        ("MODEL",LogisticRegression(random_state=0))
    ]
)



if __name__ == '__main__':

    from src.processing.data_handling import load_data,save_pipeline

    train = load_data(config.TRAIN_FILE)
    train_x = train[config.FEATURES]
    train_y = train[config.TARGET].map({'N':0,'Y':1})
    classification_pipeline.fit(train_x,train_y)
    # classification_pipeline()

