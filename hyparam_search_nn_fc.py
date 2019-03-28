#!/usr/bin/env python
# coding: utf-8

# # Master jupyter notebook for LANL - SlimBros Team

# Correctly predicting earthquakes is very important for preventing deaths and damage to infrastructure. In this competition we try to predict time left to the next laboratory earthquake based on seismic signal data. Training data represents one huge signal, but in test data we have many separate chunks, for each of which we need to predict time to failure.

# ## Preliminaries
# Let's import everything we need:

# In[1]:


import gc
import os
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

from tqdm import tqdm_notebook
from catboost import CatBoostRegressor

#from keras import layers, Sequential, callbacks, backend

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

# In[2]:


compute_features = False 
# The computed features are saved in an hdf file along with the time_to_failure to 
# save the time spend reading the training data and the feature computation
#train_data_format = 'csv'
train_data_format = 'feather'


# ## Training data

# In[3]:


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


# In[4]:


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

# In[5]:


saved_files_present = (os.path.isfile('../tmp_results/X_tr.hdf') and 
                       os.path.isfile('../tmp_results/X_test.hdf') and 
                       os.path.isfile('../tmp_results/y_tr.hdf') )


# In[6]:


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

# In[7]:


alldata = pd.concat([X_tr, X_test])
scaler = StandardScaler()
alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)
X_train_scaled = alldata[:X_tr.shape[0]]
X_test_scaled = alldata[X_tr.shape[0]:]


# ## Building models

# In[8]:


def train_model(X, X_test, y, folds, params=None, model_type='lgb',
                model=None, show_scatter=False):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    n_fold = folds.get_n_splits()
    
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'nn':
            dropout = 0.7
            num_layers = 2
            num_neurons = 128
            model = Sequential()
            model.add(layers.Dense(1024, input_dim=216, activation=tf.nn.relu))
            model.add(layers.Dropout(dropout))
            for l in range(num_layers):
                model.add(layers.Dense(num_neurons, activation=tf.nn.relu))
                model.add(layers.Dropout(dropout))
            model.add(layers.Dense(1))
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
            EPOCHS = 1000
            early_stop = callbacks.EarlyStopping(monitor='mean_absolute_error', patience=100)

            history = model.fit(
                X_train,
                y_train,
                epochs=EPOCHS,
                validation_data = (X_valid, y_valid), 
                verbose=0,
                callbacks=[early_stop, PrintDot()])
            hist = pd.DataFrame(history.history)
            val_score = hist['val_mean_absolute_error'].iloc[-1]
            print(f'val_score={val_score}')
            plot_history(history)
        
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            y_pred = model.predict(X_test).reshape(-1,)

            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = 32)
            model.fit(X_train, y_train, 
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric='mae',
                      verbose=10000,
                      early_stopping_rounds=2000)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data,
                              num_boost_round=20000,
                              evals=watchlist,
                              early_stopping_rounds=200,
                              verbose_eval=500,
                              params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns),
                                   ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            y_pred = model.predict(X_test).reshape(-1,)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', task_type='GPU', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        if model_type == 'gdi':
            y_pred_valid = gpi(X_valid).values
            y_pred = gpi(X_test).values
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))
        
        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance['feature'] = X.columns
            fold_importance['importance'] = model.feature_importances_
            fold_importance['fold'] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    if show_scatter:
        fig, axis = plt.subplots(1, 2, figsize=(12,5))
        ax1, ax2 = axis
        ax1.set_xlabel('actual')
        ax1.set_ylabel('predicted')
        ax2.set_xlabel('train index')
        ax2.set_ylabel('time to failure')
        
        ax1.scatter(y, oof, color='brown')
        ax1.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)], color='blue')

        ax2.plot(y, color='blue', label='y_train')
        ax2.plot(oof, color='orange')
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance['importance'] /= n_fold
        return oof, prediction, np.mean(scores), np.std(scores), feature_importance
    else:
        return oof, prediction, np.mean(scores), np.std(scores)


# In[9]:


n_fold = 5
folds_models = KFold(n_splits=n_fold, shuffle=True, random_state=11)

# ### NeuralNet
# This is a simple FC model mimiquing the previous genetic algorithm.

# In[10]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        else:
            print('.', end='')


# In[11]:


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
    plt.grid()
    plt.show()


# In[12]:


param_grid = {
    'dropout': [0.6, 0.65, 0.7, 0.75, 0.8],
    'num_layers' : [1, 2, 3, 4],
    'num_neurons' : [30, 45, 60, 75, 90, 105, 120, 135]
}
grid = ParameterGrid(param_grid)
results_nn = pd.DataFrame(columns=['dropout', 'num_layers', 'num_neurons', 'val_score'])

for params in tqdm_notebook(grid):
    dropout = params['dropout']
    num_layers = params['num_layers']
    num_neurons = params['num_neurons']
    print(f"Training model with {params}")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=216, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    for l in range(num_layers):
        model.add(tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    EPOCHS = 1000
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=100)
    history = model.fit(
        X_train_scaled,
        y_tr,
        epochs=EPOCHS,
        validation_split=0.5,
        verbose=0,
        callbacks=[early_stop, PrintDot()])
    hist = pd.DataFrame(history.history)
    val_score = hist['val_mean_absolute_error'].iloc[-1]
    print(f'val_score={val_score}')
    plot_history(history)

    results_nn = results_nn.append({'dropout': dropout,
                                    'num_layers' : num_layers,
                                    'num_neurons' : num_neurons,
                                    'val_score' : val_score},
                                   ignore_index=True)
    

results_nn.to_csv(f"../output/results_nn_{str(datetime.datetime.now())}.csv")
