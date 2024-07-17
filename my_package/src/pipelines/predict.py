import pandas as pd
import numpy as np
from my_package.src.processing.data_handling import load_pipeline,load_data
from my_package.configs import config




classification_model = load_pipeline()



def generate_predictions_2(path):
    data = load_data(path)
    pred = classification_model.predict(data[config.FEATURES])
    return pred

def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_model.predict(data[config.FEATURES])
    output = np.where(pred==1,'Y','N')
    result = {"prediction":output}
    return result


if __name__ == "__main__":
    predictions_train = generate_predictions_2(config.TRAIN_FILE)
    print(predictions_train)
    predictions_test = generate_predictions_2(config.TEST_FILE)
    print(predictions_test)
