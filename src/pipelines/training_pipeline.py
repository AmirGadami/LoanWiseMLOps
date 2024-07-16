from configs import config
import src.pipelines.pipeline as pipe
import src.processing.preprocessing as pp
from src.processing.data_handling import load_data,save_pipeline
import numpy as np
import pandas as pd
import sys


def training_pipline():
    train = load_data(config.TRAIN_FILE)
    train_x = train[config.FEATURES]
    train_y = train[config.TARGET].map({'N':0,'Y':1})
    pipe.classification_pipeline.fit(train_x,train_y)
    save_pipeline(pipe.classification_pipeline)



if __name__ == '__main__':
    training_pipline()





