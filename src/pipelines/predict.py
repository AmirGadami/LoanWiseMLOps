import pandas as pd
import numpy as np
from src.processing.data_handling import load_pipeline,load_data
import joblib
from configs import config





classification_model = load_pipeline()



def generate_predictions(path):
    data = load_data(path)
    pred = classification_model.predict(data[config.FEATURES])
    return pred




if __name__ == "__main__":
    predictions_train = generate_predictions(config.TRAIN_FILE)
    print(predictions_train)
    predictions_test = generate_predictions(config.TEST_FILE)
    print(predictions_test)
