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

                feature = np.asarray(data[start_index:start_index+150000])
                feature = np.expand_dims(feature, -1)
                # TODO: Review properly which label to use
                #print("Start index:", start_index)
                train_y = labels[start_index+150000-1]
                minibatch.append(feature)
                minilabels.append(train_y)
            #print("Batch labels: ", minilabels)
        else:
            for i in range(self.batch_size):
                start_index = int(self.validation_split[(self.batch_size*step)+i]*150000)#*self.seq_length)

                val_batch = np.asarray(data[start_index:start_index+150000])
                val_batch = np.expand_dims(val_batch, -1)
                val_y = labels[start_index+150000-1]

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
            self.ft = self.create_cnn1d(x, bs, is_training)
            self.y = self.create_lstmmodel(self.ft, bs, is_training)
        else:
            print("Error creating the model")

    def create_cnn1d(self, x, bs, is_traning):
        # (batch, 150000, 9) -> (batch, 750000, 18)
        conv1 = tf.layers.conv1d(inputs=x, filters=18, kernel_size=4, strides=2, padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
     
        # (batch, 750000, 18) -> (batch, 37500, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=4, strides=2,padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
     
        # (batch, 37500, 36) -> (batch, 18750, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=4, strides=2,padding='same', activation = tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
        # (batch, 18750, 72) -> (batch, 9375, 144)
        conv4 =tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=4, strides=2,padding='same', activation = tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
        
        conv5 =tf.layers.conv1d(inputs=max_pool_4, filters=288, kernel_size=4, strides=2,padding='same', activation = tf.nn.relu)
        max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')
        
            # Flatten and add dropout
        flat = tf.reshape(max_pool_5, (-1, 147*288))
     
        # Predictions
        fc1 = tf.layers.dense(flat, 1)
        print(conv1, max_pool_1)
        print(conv2,max_pool_2)
        print(conv3,max_pool_3)
        print(conv4, max_pool_4)
        print(conv5, max_pool_5)
        print(tf.shape(max_pool_4)[1])
        print(fc1)
        
        return max_pool_5

    def create_lstmmodel(self,x, bs, is_training):
        #with tf.variable_scope('SinglLSTM'):
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE):
            print("\n\nShape of LSTM")
            # print("Input X shape: ", x.get_shape())
            # # Construct the LSTM inputs and LSTM cells
            # lstm_in = tf.transpose(x, [1,0,2]) # reshape into (seq_len, N, channels)
            # print("Transpose: ", lstm_in.get_shape())
            # lstm_in = tf.reshape(lstm_in, [-1, 288]) # Now (seq_len*N, n_channels)
            # print("Reshape: ", lstm_in.get_shape())
            # lstm_in = tf.split(lstm_in, 147, 0)
            # print("Split: ", len(lstm_in))

            inputs_unstack = tf.unstack(x, axis=1)
            print("Unstack shape: ", inputs_unstack)
            fc7_out = []
            for i in inputs_unstack:
                flattened = tf.reshape(i,[-1, 288])
                fc7_out.append(flattened)
    
            aux = np.array(fc7_out)
            print(aux.shape)
            
            lstm_layer1 = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1)
            lstm_layer2 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/2, forget_bias=1)
            lstm_layer3 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/4, forget_bias=1)
            lstm_layer4 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/8, forget_bias=1)
            lstm_layer5 = tf.nn.rnn_cell.LSTMCell(self.lstm_units/16, forget_bias=1)
            

            cells = [lstm_layer1, lstm_layer2, lstm_layer3, lstm_layer4, lstm_layer5]
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)
            initial_state = stacked_lstm.zero_state(bs, tf.float32)
            
            if is_training is not None:
                trainable_bool = True
              
            else:
                trainable_bool = False
                
            wd1 = tf.get_variable("wd1", [self.lstm_units/16, 8], initializer=tf.contrib.layers.xavier_initializer())
            bd1 = tf.get_variable("bd1", [8], initializer=tf.contrib.layers.xavier_initializer())
            
            wd2 = tf.get_variable("wd2", [8, 1], initializer=tf.contrib.layers.xavier_initializer())
            bd2 = tf.get_variable("bd2", [1], initializer=tf.contrib.layers.xavier_initializer())
                
            lstm_outputs, initial_state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=initial_state, time_major=False)
            # Before transpose, lstm_outputs.get_shape() = (batch_size, num_steps, lstm_size)
            # After transpose, lstm_outputs.get_shape() = (num_steps, batch_size, lstm_size)
            output = tf.transpose(lstm_outputs, [1, 0, 2])
            print("Output: ", output)
            
            # last.get_shape() = (batch_size, lstm_size)
            last = tf.gather(output, int(output.get_shape()[0]) - 1)
            print("Last: ", last)
            
            fc1 = tf.matmul(last, wd1) + bd1
            relu1 = tf.nn.leaky_relu(fc1)
            #relu1 = tf.layers.dropout(relu1,training = trainable_bool)
            out = tf.matmul(relu1, wd2) + bd2
            print("Out shape: ", out.get_shape())
            return out

#Seeds
np.random.seed(1234)
tf.set_random_seed(1234)

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
batch_size = 16
n_features = 4 #72 #94
starter_learning_rate = 0.001
epochs = 300




print("Loading data")
train = pd.read_csv(TRAIN_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


print("Data with shape {} has the following type of data:".format(train.shape))



dataset_split = np.floor(train.shape[0]/150000) #Number of samples that fit length of 150.000
acoustic_data = train.acoustic_data.values
print(acoustic_data.shape)
del(train['acoustic_data'])
ttf = train.time_to_failure.values
print(ttf.shape)
del(train)

print("Adding additional data")
acDataStd = np.empty((0,))
acTime = np.empty((0,))
events = 297

for i in range(events):
    a = np.load(osp.join(TRAIN_PATH_ADDITIONAL, f"earthquake_{i:03d}.npz"))['acoustic_data'] 
    t = np.load(osp.join(TRAIN_PATH_ADDITIONAL, f"earthquake_{i:03d}.npz"))['ttf'] 
    acoustic_data = np.hstack([acoustic_data, a.std(axis=1)])
    ttf = np.hstack([ttf, t])



dataset = Dataset_Manager(dataset_split, TARIN_VAL_SPLIT, seq_length, batch_size, ttf)

# Model
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 150000, 1])
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
    base_loss = tf.losses.absolute_difference(y, score)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")
    out_loss = tf.reduce_mean(loss)
    tf.summary.scalar('Absolute_error', out_loss)

with tf.name_scope("mean_absolute_error"):
    eval_metrics_ops = tf.metrics.mean_absolute_error(y,score)

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





# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer_train = tf.summary.FileWriter(filewriter_path + '/train', filename_suffix = "train")
writer_test = tf.summary.FileWriter(filewriter_path + '/validation', filename_suffix = "val")

# Initialize a saver for store model checkpoints
saver = tf.train.Saver(save_relative_paths=True)

min_val_los=1000
n_steps_overfitting = 0
#with tf.Session() as sess:
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
        
        print("Epoch TRAINING done! with MSE of {} on epoch {} with gs of {}".format(np.asarray(mse_training).mean(), epoch, gs))
        
        for i in range(total_batches_test):
            batch_test_x, batch_test_y = dataset.get_next_batch(acoustic_data, ttf, False, i)
            batch_test_y = np.expand_dims(batch_test_y, -1)
            s, mse_val, result = sess.run([merged_summary, eval_metrics_ops, score],
                                             feed_dict={x: batch_test_x, y: batch_test_y, is_training: None, batch_placeholder: batch_size})
            #TODO: Discober how to add mse_val and make the avg
            #mse_total += mse_val
            #print("{} Saving checkpoint of model...".format(datetime.now()))
            # save checkpoint of the model for each epoch
            checkpoint_name = os.path.join(checkpoint_path,
                                               'model_epoch' + str(checkpoints) + '.ckpt')
            #save_path = saver.save(sess, checkpoint_name)
            mse_validation.append(mse_val[0])
    
        print("VALIDATION DONE! with MSE of {}".format(np.asarray(mse_validation).mean()))
        if (np.asarray(mse_validation).mean() < min_val_los):
            min_val_los = np.asarray(mse_validation).mean()
            n_steps_overfitting = 0
        elif(np.asarray(mse_validation).mean() - min_val_los > min_val_los*0.05):
            n_steps_overfitting += 1
            if n_steps_overfitting > 3:
                print("System is overfitting during {} epochs. Early stop.".format(n_steps_overfitting))
                break
    print("LETS TEST")

    del(ttf)
    del(acoustic_data)

    submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})
    # Load each test data, create the feature matrix, get numeric prediction
    for i, seg_id in enumerate(submission.index):
      #  print(i)
        batch_test_submission = []
        seg = pd.read_csv('../input/test/' + seg_id + '.csv')
        x_test = seg['acoustic_data'].values

        batch_test_submission = np.asarray(x_test)
        batch_test_submission = np.expand_dims(batch_test_submission, 0)
        batch_test_submission = np.expand_dims(batch_test_submission, -1)
        
        result = sess.run([score],feed_dict={x: batch_test_submission, is_training: None, batch_placeholder: 1})
        submission.time_to_failure[i] = np.abs(result[0][0][0])
    
    submission.head()
    # Save
    submission.to_csv('submission_abs.csv')
    print(submission)
    print("FINISHED! :)")
