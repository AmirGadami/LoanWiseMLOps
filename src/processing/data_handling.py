import os
import pandas as pd
import joblib
from configs import config

def load_data(file_path):
    # filePath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(file_path)
    return _data


def save_pipeline(pipline_to_save):
    piplinePath = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipline_to_save,piplinePath)
    print(f"The model has been saved under the name {config.MODEL_NAME}")


def load_pipeline():
    piplinePath = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    model_loaded = joblib.load(piplinePath)
    print("The model has been loaded")
    return model_loaded


