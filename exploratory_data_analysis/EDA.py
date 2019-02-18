import numpy as np
import pandas as pd
import utils





TRAIN_PATH = "/home/guillem/Downloads/LANL-Earthquake-Prediction/train.csv"

print("Loading data")
train = pd.read_csv(TRAIN_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

print("Data with shape {} has the following type of data:".format(train.shape))
print(train.head())

#Plotting data with a downsample of 100 rows
utils.plot_data(train['acoustic_data'].values[::100], train['time_to_failure'].values[::100])

#Plotting data first 150000 samples
utils.plot_data(train['acoustic_data'].values[:150000], train['time_to_failure'].values[:150000])

#Plotting data first 2% of data
utils.plot_data(train['acoustic_data'].values[:12582910], train['time_to_failure'].values[:12582910])


