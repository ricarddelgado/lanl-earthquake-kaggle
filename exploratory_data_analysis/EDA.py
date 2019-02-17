import numpy as np
import pandas as pd

TRAIN_PATH = "/home/guillem/Downloads/LANL-Earthquake-Prediction/train.csv"

print("Loading data")
train = pd.read_csv(TRAIN_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

print("Data with shape {} has the following type of data:".format(train.shape))
print(train.head())
