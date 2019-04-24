#!/usr/bin/env python
# coding: utf-8

# # Master jupyter notebook for LANL - SlimBros Team

# Correctly predicting earthquakes is very important for preventing deaths and damage to infrastructure. In this competition we try to predict time left to the next laboratory earthquake based on seismic signal data. Training data represents one huge signal, but in test data we have many separate chunks, for each of which we need to predict time to failure.

# ## Preliminaries
# Let's import everything we need:

# In[1]:


import gc
import os
import csv
import time
import random
import datetime
import warnings
import feather

import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import lightgbm as lgb
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GridSearchCV, cross_val_score
from utils import generate_segment_start_ids, compare_methods
from features import gpi, create_all_features

#Configure the environment
pd.options.display.precision = 15
warnings.filterwarnings('ignore')
random.seed(1013)


# Load/compute the necessary features
compute_features = False 
train_data_format = 'feather'


# ## Training data

def load_train_data(file_format):
    """Load the training dataset."""
    print(f"Loading data from {file_format} file:", end="")
    if file_format.lower() == 'feather':
        train_df = feather.read_dataframe('../input/train.feather')
    else:
        train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16,
                                                            'time_to_failure': np.float32})
        feather.write_dataframe(train_df, '../input/train.feather')
    print("Done")
    return train_df

train = load_train_data(train_data_format)

# ## Feature generation
# - Usual aggregations: mean, std, min and max;
# - Average difference between the consequitive values in absolute and percent values;
# - Absolute min and max vallues;
# - Aforementioned aggregations for first and last 10000 and 50000 values - I think these data should be useful;
# - Max value to min value and their differencem also count of values bigger that 500 (arbitrary threshold);
# - Quantile features from this kernel: https://www.kaggle.com/andrekos/basic-feature-benchmark-with-quantiles
# - Trend features from this kernel: https://www.kaggle.com/jsaguiar/baseline-with-abs-and-trend-features
# - Rolling features from this kernel: https://www.kaggle.com/wimwim/rolling-quantiles

saved_files_present = (os.path.isfile('../tmp_results/X_tr.hdf') and 
                       os.path.isfile('../tmp_results/X_test.hdf') and 
                       os.path.isfile('../tmp_results/y_tr.hdf') )


if (not compute_features) and saved_files_present:
    print(f"Reading hdf files:", end="")
    X_tr = pd.read_hdf('../tmp_results/X_tr.hdf', 'data')
    X_test = pd.read_hdf('../tmp_results/X_test.hdf', 'data')
    y_tr = pd.read_hdf('../tmp_results/y_tr.hdf', 'data')  
    print("Done")
else:
    fs = 4000000 #Sampling frequency of the raw signal

    #Compute features for the training data
    segment_size = 150000
    segment_start_ids = generate_segment_start_ids('uniform_no_jump', segment_size, train)
    X_tr = pd.DataFrame(index=range(len(segment_start_ids)), dtype=np.float64)
    y_tr = pd.DataFrame(index=range(len(segment_start_ids)), dtype=np.float64, columns=['time_to_failure'])
    for idx in tqdm_notebook(range(len(segment_start_ids))):        
        seg_id = segment_start_ids[idx]
        seg = train.iloc[seg_id:seg_id + segment_size]
        create_all_features(idx, seg, X_tr, fs)
        y_tr.loc[idx, 'time_to_failure'] = seg['time_to_failure'].values[-1]
    # Sanity check
    means_dict = {}
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            print(col)
            mean_value = X_tr.loc[X_tr[col] != -np.inf, col].mean()
            X_tr.loc[X_tr[col] == -np.inf, col] = mean_value
            X_tr[col] = X_tr[col].fillna(mean_value)
            means_dict[col] = mean_value

    #Compute features for the test data
    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)
    for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
        seg = pd.read_csv('../input/test/' + seg_id + '.csv')
        create_all_features(seg_id, seg, X_test, fs)

    # Sanity check
    for col in X_test.columns:
        if X_test[col].isnull().any():
            X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
            X_test[col] = X_test[col].fillna(means_dict[col])
            
    X_tr.to_hdf('../tmp_results/X_tr.hdf', 'data')
    X_test.to_hdf('../tmp_results/X_test.hdf', 'data')
    y_tr.to_hdf('../tmp_results/y_tr.hdf', 'data')
    
    del segment_start_ids
    del means_dict
    del submission
    
    print("Done")


# ## Scale data
alldata = pd.concat([X_tr, X_test])
scaler = StandardScaler()
alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)
X_train_scaled = alldata[:X_tr.shape[0]]
X_test_scaled = alldata[X_tr.shape[0]:]

# ### NeuralNet
# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        else:
            print('.', end='')

X_train_nn = X_train_scaled

num_random_draws = 2000
tiny_nn_results_file = "../output/search_all_features_nn_5x1x1_tanh_linear/small_nn_search_{}.csv".format(str(datetime.datetime.now()))

headers = ['f1','f2','f3','f4','f5','val_score']

with open(tiny_nn_results_file, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(headers)
writeFile.close()

for i in tqdm(range(num_random_draws)):
    X_train_nn_sampled = X_train_nn.sample(5, axis=1) #pick 5 random columns
    
    line = X_train_nn_sampled.columns.tolist()

    activation_function = 'tanh' #tf.nn.tanh
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1,
                                    input_dim=X_train_nn_sampled.shape[-1],
                                    activation=activation_function))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    EPOCHS = 200
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=100)
    history = model.fit(
        X_train_nn_sampled,
        y_tr,
        validation_split=0.2,
        epochs=EPOCHS,
        shuffle=True,
        verbose=0,
        callbacks=[early_stop])
    hist = pd.DataFrame(history.history)
    val_score = hist['val_mean_absolute_error'].iloc[-1]
    line.append(val_score)

    #Housekeeping
    del model
    tf.keras.backend.clear_session()
    
    with open(tiny_nn_results_file, 'a') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(line)

    writeFile.close()