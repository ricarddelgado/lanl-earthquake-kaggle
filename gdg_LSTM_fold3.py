#!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
#!unzip ngrok-stable-linux-amd64.zip
#get_ipython().system_raw('./ngrok http 6006 &')
#! curl -s http://localhost:4040/api/tunnels | python3 -c \
#    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

#TODO list
# - Normalize features. Mean substraction and division by std ok
# - Find a way to get best features
# - Maybe BN??

from __future__ import division
#This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by th/e kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
import os.path as osp
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
import math
import random
#from scipy import stats
import scipy
from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate, nonzero
from matplotlib.mlab import find
from statsmodels import robust
from scipy.signal import blackmanharris, fftconvolve
from scipy.signal import hilbert, hann, convolve
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

class Dataset_Manager():
    def __init__(self, dataset_split, TRAIN_VAL_SPLIT, sequence_length, batch_size, ttf):
        self.downsample = int(150000/sequence_length)
        #index = np.arange(dataset_split)#TODO: This index should be multiplied by 150000
        index = self.generate_segment_start_ids(dataset_split, 'uniform_no_jump', ttf)
        np.random.shuffle(index)
        
        print("Sequence lentgh of {} where the original length of 150.000 has been downsampled of: {}".format(sequence_length, self.downsample))
        train_samples = int(round(len(index)*TRAIN_VAL_SPLIT))
        validation_samples = int(len(index)-train_samples)
        print("The dataset contains {} train samples and {} validation samples which is a {} ratio".format(
            train_samples, validation_samples, TRAIN_VAL_SPLIT))



        self.validation_split = index[0:validation_samples]
        self.train_split = index[validation_samples+1:validation_samples+train_samples]
        self.seq_length = sequence_length
        print("Len split train", len(self.train_split))
        print("Len split val", len(self.validation_split))
        self.total_batches_train = math.floor(len(self.train_split)/batch_size)
        self.total_batches_validation = math.floor(len(self.validation_split)/batch_size)
        self.batch_size = batch_size
        
        print("Total batches train: ", self.total_batches_train)
        print("Total batches val: ", self.total_batches_validation)
    
    def generate_segment_start_ids(self, dataset_split, sampling_method, train):
        if sampling_method == 'uniform_no_jump':
            # With this approach we obtain 4178 segments (99.5% of 'uniform')
            num_segments = int(dataset_split)
            print("Number of splits {}".format(num_segments))
            time_to_failure_jumps = np.diff(train)
            num_good_segments_found = 0
            segment_start_ids = []
            for i in range(num_segments):
                idx = i * 150000
                # Detect if there is a discontinuity on the time_to_failure signal within the segment
                max_jump = np.max(time_to_failure_jumps[idx:idx + 150000])
                if max_jump < 5:
                    segment_start_ids.append(i)
                    num_good_segments_found += 1
                else:
                    print(f'Rejected candidate segment since max_jump={max_jump}')
            segment_start_ids.sort()
        else:
            print("Not a sampling method")
            exit(-1)
        return segment_start_ids
    
    def freq_from_crossings(self, sig, fs):
        """
        Estimates frequency by counting zero crossings
        """
    
        # Find all indices right before a rising-edge zero crossing
        idx = nonzero((sig[1:] >= 0) & (sig[:-1] < 0))
        
        # More accurate, using linear interpolation to find intersample 
        # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
        crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in idx]
        
        diff_crossings = np.diff(crossings[0])
        min_diff_crossings = np.min(diff_crossings)
        max_diff_crossings = np.max(diff_crossings)
        mean_diff_crossings = np.mean(diff_crossings)
        median_diff_crossings = np.median(diff_crossings) 
    
        mean_freq = fs / mean_diff_crossings
        median_freq = fs / median_diff_crossings
    
        mean_freq = fs / mean(diff(crossings[0]))
        median_freq = fs / np.median(diff(crossings[0]))
    
        
        return median_freq
    
    def add_trend_feature(self, arr, abs_values=False):
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]

    def classic_sta_lta(self, x, length_sta, length_lta, show=False, method='modified'):
        if method == 'modified':
            x_abs = np.abs(x)
            # Convert to float
            x_abs = np.require(x_abs, dtype=np.float)
            # Compute the STA and the LTA
            sta = np.cumsum(x_abs)
            sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
            sta = sta[length_sta - 1:] / length_sta
            sta = sta[:-(length_lta-length_sta)]
            lta = x_abs.copy()
            lta = np.cumsum(lta)
            lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
            lta = lta[length_lta - 1:] / length_lta
    
            ratio = sta / lta
            return ratio
    
    def change_rate(self, x, method='original'):
        if method == 'original':
            rate = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
        if method == 'modified':
            change = (np.diff(x) / x[:-1])
            change = change[np.nonzero(change)[0]]
            change = change[~np.isnan(change)]
            change = change[change != -np.inf]
            change = change[change != np.inf]
            rate = np.mean(change)
        return rate


    def create_features(self, xc, train_batch):
        fs = 4000000
        #train_batch.append(xc)

        # Generic stats
        train_batch.append(xc.mean())
        
        # #### From here -->>
        train_batch.append(xc.std())
        train_batch.append(xc.max())
        train_batch.append(xc.min())
        
        # # rdg: mean_change_abs corrected
        #train_batch.append(np.mean(np.diff(xc)))
        #train_batch.append(np.mean(np.abs(np.diff(xc))))
        # train_batch.append(self.change_rate(xc, method='original'))
        #train_batch.append(self.change_rate(xc, method='modified'))
        #train_batch.append(np.abs(xc).max())
        #train_batch.append(np.abs(xc).min())

        
        #train_batch.append(xc.max() / np.abs(xc.min()))
        #train_batch.append(xc.max() - np.abs(xc.min()))
        #train_batch.append(len(xc[np.abs(xc) > 500]))
        # train_batch.append(xc.sum())
        
        # train_batch.append(self.change_rate(xc[:500], method='original'))
        # train_batch.append(self.change_rate(xc[-500:], method='original'))
        # train_batch.append(self.change_rate(xc[:1000], method='original'))
        # train_batch.append(self.change_rate(xc[-1000:], method='original'))
        
        # train_batch.append(self.change_rate(xc[:500], method='modified'))
        # train_batch.append(self.change_rate(xc[-500:], method='modified'))
        # train_batch.append(self.change_rate(xc[:1000], method='modified'))
        # train_batch.append(self.change_rate(xc[-1000:], method='modified'))
        
        #train_batch.append(np.quantile(xc, 0.95))
        #train_batch.append(np.quantile(xc, 0.99))
        #train_batch.append(np.quantile(xc, 0.05))
        #train_batch.append(np.quantile(xc, 0.01))
        
        # train_batch.append(np.quantile(np.abs(xc), 0.95))
        # train_batch.append(np.quantile(np.abs(xc), 0.99))
        # train_batch.append(np.quantile(np.abs(xc), 0.05))
        # train_batch.append(np.quantile(np.abs(xc), 0.01))
        
        # train_batch.append(self.add_trend_feature(xc))
        # train_batch.append(self.add_trend_feature(xc, abs_values=True))
        # train_batch.append(np.abs(xc).mean())
        # train_batch.append(np.abs(xc).std())
        
        # train_batch.append(robust.mad(xc))
        # train_batch.append(scipy.stats.kurtosis(xc))
        # train_batch.append(scipy.stats.skew(xc))
        # train_batch.append(np.median(xc))
        # # ### From here --->
        
        
        # train_batch.append(np.abs(hilbert(xc)).mean())
        # train_batch.append((convolve(xc, hann(150), mode='same') / sum(hann(150))).mean())    
        
        # sta_lta_method = 'modified'
        # classic_sta_lta1 = self.classic_sta_lta(xc, 50, 100, method=sta_lta_method)
        # classic_sta_lta2 = self.classic_sta_lta(xc, 500, 1000, method=sta_lta_method)
        # classic_sta_lta3 = self.classic_sta_lta(xc, 10, 500, method=sta_lta_method)
        # classic_sta_lta4 = self.classic_sta_lta(xc, 100, 250, method=sta_lta_method)
        # classic_sta_lta5 = self.classic_sta_lta(xc, 200, 500, method=sta_lta_method)
        # classic_sta_lta6 = self.classic_sta_lta(xc, 100, 500, method=sta_lta_method)
        # classic_sta_lta7 = self.classic_sta_lta(xc, 333, 666, method=sta_lta_method)
        # classic_sta_lta8 = self.classic_sta_lta(xc, 400, 1000, method=sta_lta_method)
        

        # train_batch.append(classic_sta_lta1.mean())
        # train_batch.append(classic_sta_lta2.mean())
        # train_batch.append(classic_sta_lta3.mean())
        # train_batch.append(classic_sta_lta4.mean())
        # train_batch.append(classic_sta_lta5.mean())
        # train_batch.append(classic_sta_lta6.mean())
        # train_batch.append(classic_sta_lta7.mean())
        # train_batch.append(classic_sta_lta8.mean())
    
        # train_batch.append(np.quantile(classic_sta_lta1, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta2, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta3, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta4, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta5, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta6, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta7, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta8, 0.95))  
    
        # train_batch.append(np.quantile(classic_sta_lta1, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta2, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta3, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta4, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta5, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta6, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta7, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta8, 0.05))
    
        # train_batch.append(np.subtract(*np.percentile(xc, [75, 25])))
        # train_batch.append(np.quantile(xc, 0.999))
        # train_batch.append(np.quantile(xc, 0.001))
        # train_batch.append(scipy.stats.trim_mean(xc, 0.1))
    
        # # # rdg: The frequency features are new
        # train_batch.append(self.freq_from_crossings(xc[:5000], fs))
        # train_batch.append(self.freq_from_crossings(xc[-5000:], fs))
        # train_batch.append(self.freq_from_crossings(xc[:1000], fs))
        # train_batch.append(self.freq_from_crossings(xc[-1000:], fs))


    def generate_batch(self):

        np.random.shuffle(self.train_split)

        np.random.shuffle(self.validation_split)

    def get_next_batch(self, data, labels, train, step):
        minibatch = []
        minilabels = []
        if train:
            for i in range(self.batch_size):
                start_index = int(self.train_split[(self.batch_size*step)+i]*150000)#self.seq_length)
                if(bool(random.getrandbits(1))):
                    shifted = start_index + random.randint(1,75001)
                    if shifted < 628995480:
                        start_index = shifted
                train_batch = []
                for mini_seq in range (int(np.floor(150000/self.downsample))):
                    mini_batch_seq = []
                    self.create_features(data[start_index+(self.downsample*mini_seq):(start_index+(self.downsample*mini_seq)+self.downsample)], mini_batch_seq)
                    #mini_batch_seq = (mini_batch_seq -  np.mean(mini_batch_seq, axis=0, keepdims=True)) / np.mean(mini_batch_seq, axis=0, keepdims=True)
                    if(mini_seq == 0):
                        train_batch = mini_batch_seq
                    else:
                        train_batch = np.vstack((train_batch,mini_batch_seq))

                feature = np.asarray(train_batch)
                # TODO: Review properly which label to use
                #print("Start index:", start_index)
                train_y = labels[start_index+150000-1]
                minibatch.append(feature)
                minilabels.append(train_y)
            #print("Batch labels: ", minilabels)
        else:
            for i in range(self.batch_size):
                start_index = int(self.validation_split[(self.batch_size*step)+i]*150000)#*self.seq_length)
                val_batch = []
                for mini_seq in range (int(np.floor(150000/self.downsample))):
                    mini_batch_seq_val = []
                    self.create_features(data[start_index+(self.downsample*mini_seq):(start_index+(self.downsample*mini_seq)+self.downsample)], mini_batch_seq_val)
                    #mini_batch_seq_val = (mini_batch_seq_val - np.mean(mini_batch_seq_val, axis=0, keepdims=True)) / np.mean(mini_batch_seq_val, axis=0, keepdims=True)
                    if(mini_seq == 0):
                        val_batch = mini_batch_seq_val
                    else:
                        val_batch = np.vstack((val_batch,mini_batch_seq_val))
    
                feature_val = np.asarray(val_batch)
                val_y = labels[start_index+150000-1]
                #val_batch = np.expand_dims(val_batch, -1)
                minibatch.append(val_batch)
                minilabels.append(val_y)
        return minibatch, minilabels
        
class model(object):

    def __init__(self, x, bs, seq_lenght, skip_layer, is_training, whichmodel, n_features, batch_size):
        # define constants
        # unrolled through 49 time steps
        self.seq_lenght = seq_lenght
        self.n_features = n_features
        # hidden LSTM units
        self.lstm_units = 128

        self.batch_size = batch_size

        self.SKIP_LAYER = skip_layer

        self.is_training = is_training


        if whichmodel == 'LSTM':
            self.y = self.create_lstmmodel(x, bs, is_training)
        else:
            print("Error creating the model")

    def cnn_extractor(self, x, bs):
        feat = tf.layers.Conv1D()
        feat = tf.layers.Conv1D()

    def create_lstmmodel(self,x, bs, is_training):
        #with tf.variable_scope('SinglLSTM'):
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE):
            print("\n\nShape of LSTM")
            
            print("Input X shape: ", x.get_shape())
            #print(tf.shape(x))
            #print("------")
            
            
            inputs_unstack = tf.unstack(x, axis=1)
            #print(len(inputs_unstack))
            print("Unstack shape: ", inputs_unstack)
            #print("------")
            fc7_out = []
            for i in inputs_unstack:
                flattened = tf.reshape(i,[-1, self.n_features])
                fc7_out.append(flattened)
    
            aux = np.array(fc7_out)
            #print(len(fc7_out))
            #print(fc7_out[1].get_shape())
            print(aux.shape)
            #print("------")
    
    
    
            # lstm_layer = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1)
            # lstm_outputs, _ = rnn.static_rnn(lstm_layer, fc7_out, dtype="float32")
            

            
            
            lstm_layer1 = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1)
            lstm_layer2 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/2, forget_bias=1)
            lstm_layer3 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/4, forget_bias=1)
            lstm_layer4 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/8, forget_bias=1)
            lstm_layer5 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/16, forget_bias=1)
            
            #num_units = [self.lstm_units, self.lstm_units/2]

            cells = [lstm_layer1, lstm_layer2, lstm_layer3, lstm_layer4, lstm_layer5]
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)
            #initial_state = stacked_lstm.zero_state(self.batch_size, tf.float32)
            initial_state = stacked_lstm.zero_state(bs, tf.float32)

            # if is_training is not None:
            #     wd1 = tf.get_variable("wd1", [self.lstm_units, 10], initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
            #     bd1 = tf.get_variable("bd1", [10], initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
                
            #     wd2 = tf.get_variable("wd2", [10, 1], initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
            #     bd2 = tf.get_variable("bd2", [1], initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
            # else:
            #     wd1 = tf.get_variable("wd2", [self.lstm_units, 1], initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
            #     bd1 = tf.get_variable("bd2", [1], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
                
            #     wd2 = tf.get_variable("wd2", [10, 1], initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
            #     bd2 = tf.get_variable("bd2", [1], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            
            
            if is_training is not None:
                trainable_bool = True
              
            else:
                trainable_bool = False
                
            wd1 = tf.get_variable("wd1", [self.lstm_units/16, 8], initializer=tf.contrib.layers.xavier_initializer())
            bd1 = tf.get_variable("bd1", [8], initializer=tf.contrib.layers.xavier_initializer())
            
            wd2 = tf.get_variable("wd2", [8, 1], initializer=tf.contrib.layers.xavier_initializer())
            bd2 = tf.get_variable("bd2", [1], initializer=tf.contrib.layers.xavier_initializer())
                
    
            # Regression
            # count = 0
            # for feature in fc7_out:
            #     lstm_outputs, state = cell(feature, state)
            #     count += 1
            #     if count == self.seq_lenght:
            #         fc1 = tf.matmul(lstm_outputs[-1], wd1) + bd1
            #         relu1 = tf.nn.leaky_relu(fc1)
            #         relu1 = tf.layers.dropout(relu1,training = trainable_bool)
            #         out = tf.matmul(relu1, wd2) + bd2
            #         print("Out shape: ", out.get_shape())
                    
            #         print("---- \n\n")
            #         #y = tf.nn.softmax(fc1)
            
            #         return out
            lstm_outputs, initial_state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=initial_state, time_major=False)
            output = tf.transpose(lstm_outputs, [1, 0, 2])
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            fc1 = tf.matmul(last, wd1) + bd1
            relu1 = tf.nn.leaky_relu(fc1)
            #relu1 = tf.layers.dropout(relu1,training = trainable_bool)
            out = tf.matmul(relu1, wd2) + bd2
            print("Out shape: ", out.get_shape())
            return out

    def load_initial_weights(self, session):

        # Load the weights into memory
        weights_dict = np.load("DEFAULT", encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

                    

def create_test_vector(xc, train_batch, dataset):
        fs = 4000000
        #train_batch.append(xc)

        # Generic stats
        train_batch.append(xc.mean())
        
        ### From here --->>
        train_batch.append(xc.std())
        train_batch.append(xc.max())
        train_batch.append(xc.min())
        
        # # rdg: mean_change_abs corrected
        #train_batch.append(np.mean(np.diff(xc)))
        #train_batch.append(np.mean(np.abs(np.diff(xc))))
        # train_batch.append(dataset.change_rate(xc, method='original'))
        #train_batch.append(dataset.change_rate(xc, method='modified'))
        #train_batch.append(np.abs(xc).max())
        #train_batch.append(np.abs(xc).min())

        
        #train_batch.append(xc.max() / np.abs(xc.min()))
        #train_batch.append(xc.max() - np.abs(xc.min()))
        #train_batch.append(len(xc[np.abs(xc) > 500]))
        #train_batch.append(xc.sum())
        
        # train_batch.append(dataset.change_rate(xc[:500], method='original'))
        # train_batch.append(dataset.change_rate(xc[-500:], method='original'))
        # train_batch.append(dataset.change_rate(xc[:1000], method='original'))
        # train_batch.append(dataset.change_rate(xc[-1000:], method='original'))
        
        # train_batch.append(dataset.change_rate(xc[:500], method='modified'))
        # train_batch.append(dataset.change_rate(xc[-500:], method='modified'))
        # train_batch.append(dataset.change_rate(xc[:1000], method='modified'))
        # train_batch.append(dataset.change_rate(xc[-1000:], method='modified'))
        
        # train_batch.append(np.quantile(xc, 0.95))
        # train_batch.append(np.quantile(xc, 0.99))
        # train_batch.append(np.quantile(xc, 0.05))
        # train_batch.append(np.quantile(xc, 0.01))
        
        # # train_batch.append(np.quantile(np.abs(xc), 0.95))
        # # train_batch.append(np.quantile(np.abs(xc), 0.99))
        # # train_batch.append(np.quantile(np.abs(xc), 0.05))
        # # train_batch.append(np.quantile(np.abs(xc), 0.01))
        
        # train_batch.append(dataset.add_trend_feature(xc))
        # train_batch.append(dataset.add_trend_feature(xc, abs_values=True))
        # train_batch.append(np.abs(xc).mean())
        # train_batch.append(np.abs(xc).std())
        
        # train_batch.append(robust.mad(xc))
        # train_batch.append(scipy.stats.kurtosis(xc))
        # train_batch.append(scipy.stats.skew(xc))
        # train_batch.append(np.median(xc))
        # ### From here --->>
        
        
        # train_batch.append(np.abs(hilbert(xc)).mean())
        # train_batch.append((convolve(xc, hann(150), mode='same') / sum(hann(150))).mean())    
        
        # sta_lta_method = 'modified'
        # classic_sta_lta1 = dataset.classic_sta_lta(xc, 50, 100, method=sta_lta_method)
        # classic_sta_lta2 = dataset.classic_sta_lta(xc, 500, 1000, method=sta_lta_method)
        # classic_sta_lta3 = dataset.classic_sta_lta(xc, 10, 500, method=sta_lta_method)
        # classic_sta_lta4 = dataset.classic_sta_lta(xc, 100, 250, method=sta_lta_method)
        # classic_sta_lta5 = dataset.classic_sta_lta(xc, 200, 500, method=sta_lta_method)
        # classic_sta_lta6 = dataset.classic_sta_lta(xc, 100, 500, method=sta_lta_method)
        # classic_sta_lta7 = dataset.classic_sta_lta(xc, 333, 666, method=sta_lta_method)
        # classic_sta_lta8 = dataset.classic_sta_lta(xc, 400, 1000, method=sta_lta_method)
        

        # train_batch.append(classic_sta_lta1.mean())
        # train_batch.append(classic_sta_lta2.mean())
        # train_batch.append(classic_sta_lta3.mean())
        # train_batch.append(classic_sta_lta4.mean())
        # train_batch.append(classic_sta_lta5.mean())
        # train_batch.append(classic_sta_lta6.mean())
        # train_batch.append(classic_sta_lta7.mean())
        # train_batch.append(classic_sta_lta8.mean())
    
        # train_batch.append(np.quantile(classic_sta_lta1, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta2, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta3, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta4, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta5, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta6, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta7, 0.95))
        # train_batch.append(np.quantile(classic_sta_lta8, 0.95))  
    
        # train_batch.append(np.quantile(classic_sta_lta1, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta2, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta3, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta4, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta5, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta6, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta7, 0.05))
        # train_batch.append(np.quantile(classic_sta_lta8, 0.05))
    
        # train_batch.append(np.subtract(*np.percentile(xc, [75, 25])))
        # train_batch.append(np.quantile(xc, 0.999))
        # train_batch.append(np.quantile(xc, 0.001))
        # train_batch.append(scipy.stats.trim_mean(xc, 0.1))
    
        # # # rdg: The frequency features are new
        # train_batch.append(dataset.freq_from_crossings(xc[:5000], fs))
        # train_batch.append(dataset.freq_from_crossings(xc[-5000:], fs))
        # train_batch.append(dataset.freq_from_crossings(xc[:1000], fs))
        # train_batch.append(dataset.freq_from_crossings(xc[-1000:], fs))

#Seeds
np.random.seed(6969)
tf.set_random_seed(6969)

CUDA_VISIBLE_DEVICES=0
config = tf.ConfigProto()

# Variables
TRAIN_PATH = "../input/train.csv"
TRAIN_PATH_ADDITIONAL = '../input/p4581'
TARIN_VAL_SPLIT = 0.75
LOG_DIR = './folder_to_save_graph_3'
filewriter_path = LOG_DIR
checkpoint_path = ""

# Local variables
seq_length = 150
batch_size = 32
n_features = 4 #72 #94
starter_learning_rate = 0.001
epochs = 400




print("Loading data")
train = pd.read_csv(TRAIN_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


print("Data with shape {} has the following type of data:".format(train.shape))



dataset_split = np.floor(train.shape[0]/150000) #Number of samples that fit length of 150.000
acoustic_data = train.acoustic_data.values
print(acoustic_data.shape)
del(train['acoustic_data'])
ttf = np.sqrt(train.time_to_failure.values)
print(ttf.shape)
del(train)

print("Adding additional data")
acDataStd = np.empty((0,))
acTime = np.empty((0,))
events = 297

#for i in range(events):
#    a = np.load(osp.join(TRAIN_PATH_ADDITIONAL, f"earthquake_{i:03d}.npz"))['acoustic_data'] 
#    t = np.load(osp.join(TRAIN_PATH_ADDITIONAL, f"earthquake_{i:03d}.npz"))['ttf'] 
#    acoustic_data = np.hstack([acoustic_data, a.std(axis=1)])
#    ttf = np.hstack([ttf, t])
    
dataset = Dataset_Manager(dataset_split, TARIN_VAL_SPLIT, seq_length, batch_size, ttf)

# Model
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, seq_length, n_features])
batch_placeholder = tf.placeholder(tf.int32, [], name='batch_size')
y = tf.placeholder(tf.float32, [None, 1])
is_training = tf.placeholder(tf.bool)

kernel = model(x, batch_placeholder, seq_length, [], is_training, 'LSTM', n_features, batch_size)
score = kernel.y
print("Score shape: ", score.get_shape())
# List of trainable variables
var_list = tf.trainable_variables()
print("----")

with tf.name_scope("cost_function"):
    print("Loading loss")
    #base_loss = tf.losses.absolute_difference(y, score)
    base_loss = tf.losses.mean_squared_error(y, score)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")
    out_loss = tf.reduce_mean(loss)
    tf.summary.scalar('Absolute_error', out_loss)

with tf.name_scope("mean_absolute_error"):
    eval_metrics_ops = tf.metrics.mean_absolute_error(y,score)
#tf.summary.scalar('MAE_metric', tf.squeeze(eval_metrics_ops))

with tf.name_scope("train"):
    # add an optimiser
    print("Loading gradients")
    global_step = tf.Variable(0, trainable=False)
    print("Global step")
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.95, staircase=True)
    print("learning rate")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(out_loss)
    print("Optimizer loading")
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    print("Grads")

#for var in var_list:
#        tf.summary.histogram(var.name, var)

for grad, var in grads:
    #print(var)
    #print(grad)
    #print("-----")
    tf.summary.histogram(var.name + '/gradient', grad)




# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer_train = tf.summary.FileWriter(filewriter_path + '/train', filename_suffix = "train")
writer_test = tf.summary.FileWriter(filewriter_path + '/validation', filename_suffix = "val")

# Initialize a saver for store model checkpoints
saver = tf.train.Saver(save_relative_paths=True, max_to_keep=100)

min_val_los=1000
n_steps_overfitting = 0
#with tf.Session() as sess:


best_epoch = 0
best_mse = 10000
plot_train_error = []
plot_val_error = []
with tf.Session(config = config) as sess:
    # initialise the variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    writer_train.add_graph(sess.graph)
    writer_test.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                     filewriter_path))
    checkpoints = 0
    for epoch in range(epochs):
        num_batches = 0
        dataset.generate_batch()
        total_batches_train = dataset.total_batches_train
        total_batches_test = dataset.total_batches_validation
        total_loss_train = 0
        total_loss_val = 0
        mse_training = []
        mse_validation = []
        for step in range(dataset.total_batches_train):
            gs = (epoch * dataset.total_batches_train) + step + 1

            batch_x, batch_y = dataset.get_next_batch(acoustic_data, ttf , True, step)
            #print("Shape X: ", np.asarray(batch_x).shape)
            batch_y = np.expand_dims(batch_y, -1)
            #print("Shape Y: ", np.asarray(batch_y).shape)
            s, _, loss_mse, base, results_train = sess.run([merged_summary, optimizer, out_loss, base_loss, score],
                               feed_dict={x: batch_x, y: batch_y, global_step: gs, is_training: True, batch_placeholder: batch_size})
                               
            mse_training.append(loss_mse)
            writer_train.add_summary(s, (epoch * total_batches_train) + step)
        
        print("Epoch TRAINING done! with MSE of {} on epoch {} with gs of {}".format(np.asarray(mse_training).mean(), epoch, gs))
        plot_train_error.append(np.asarray(mse_training).mean())
        for i in range(total_batches_test):
            batch_test_x, batch_test_y = dataset.get_next_batch(acoustic_data, ttf, False, i)
            batch_test_y = np.expand_dims(batch_test_y, -1)
            s, mse_val, result = sess.run([merged_summary, eval_metrics_ops, score],
                                             feed_dict={x: batch_test_x, y: batch_test_y, is_training: None, batch_placeholder: batch_size})
            writer_test.add_summary(s, (epoch * dataset.total_batches_validation) + i + step)
            #TODO: Discober how to add mse_val and make the avg
            #mse_total += mse_val
            #print("{} Saving checkpoint of model...".format(datetime.now()))
            # save checkpoint of the model for each epoch
            checkpoint_name = os.path.join(checkpoint_path,
                                               'model_epoch' + str(checkpoints) + '.ckpt')
            #save_path = saver.save(sess, checkpoint_name)
            mse_validation.append(mse_val[0])
            
        if np.asarray(mse_validation).mean() < best_mse:
            best_mse = np.asarray(mse_validation).mean()
            best_epoch = epoch
    
        print("VALIDATION DONE! with MSE of {}".format(np.asarray(mse_validation).mean()))
        plot_val_error.append(np.asarray(mse_validation).mean())
        
        print("Best model at epoch {} with mse of {}".format(best_epoch,best_mse))
        saver.save(sess, 'earthquake-epoch'+ str(epoch) + '.ckpt')
        if (np.asarray(mse_validation).mean() < min_val_los):
            min_val_los = np.asarray(mse_validation).mean()
            n_steps_overfitting = 0
        elif(np.asarray(mse_validation).mean() - min_val_los > min_val_los*0.01):
            n_steps_overfitting += 1
            if n_steps_overfitting > 3:
                print("System is overfitting during {} epochs. Early stop.".format(n_steps_overfitting))
                #break
    print("LETS TEST")

    del(ttf)
    del(acoustic_data)
    plt.plot(plot_train_error,color='blue', label='Train error')
    plt.plot(plot_val_error, color='red', label='Eval error')
    plt.xlabel("Epochs")
    plt.ylabel("Epochs")
    plt.legend()
    plt.savefig("MAE Loss")


    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
    # Load each test data, create the feature matrix, get numeric prediction
    for i, seg_id in enumerate(submission.index):
      #  print(i)
        batch_test_submission = []
        seg = pd.read_csv('../input/test/' + seg_id + '.csv')
        x_test = seg['acoustic_data'].values
        downsample = int(150000/seq_length)
        for mini_seq in range (int(np.floor(150000/downsample))):
            mini_batch_seq_val = []
            create_test_vector(x_test[0+(downsample*mini_seq):(0+(downsample*mini_seq)+downsample)], mini_batch_seq_val, dataset)
            #mini_batch_seq_val = (mini_batch_seq_val -  np.mean(mini_batch_seq_val, axis=0, keepdims=True)) / np.mean(mini_batch_seq_val, axis=0, keepdims=True)
            
            if(mini_seq == 0):
                batch_test_submission = mini_batch_seq_val
            else:
                batch_test_submission = np.vstack((batch_test_submission,mini_batch_seq_val))

        feature_val = np.asarray(batch_test_submission)
        #batch_test_submission.append(feature_val)
        batch_test_submission = np.asarray(batch_test_submission)
        batch_test_submission = np.expand_dims(batch_test_submission, 0)
        #print("Shape test: ", batch_test_submission.shape)
        
        result = sess.run([score],feed_dict={x: batch_test_submission, is_training: None, batch_placeholder: 1})
        submission.time_to_failure[i] = np.power(result[0][0][0],2)
    
    submission.head()
    # Save
    submission.to_csv('../output/submission_abs_fold3.csv')
    print(submission)
    print("FINISHED! :)")

