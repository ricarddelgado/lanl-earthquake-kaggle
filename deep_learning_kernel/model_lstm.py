import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn


class model(object):

    def __init__(self, x, seq_lenght, skip_layer, is_training, whichmodel):
        # define constants
        # unrolled through 49 time steps
        self.seq_lenght = seq_lenght

        # hidden LSTM units
        self.lstm_units = 512



        self.SKIP_LAYER = skip_layer

        self.is_training = is_training


        if whichmodel == 'LSTM':
            self.y = self.create_lstmmodel(x, is_training)
        else:
            print("Error creating the model")






    def create_lstmmodel(self,x, eval):

        lstm_layer = rnn.BasicLSTMCell(self.lstm_units, forget_bias=1)
        lstm_outputs, _ = rnn.static_rnn(lstm_layer, x, dtype="float32")

        wd1 = tf.get_variable("wd1", [self.lstm_units, 1], initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=eval)
        bd1 = tf.get_variable("bd1", [1], initializer=tf.contrib.layers.xavier_initializer(), trainable=eval)

        # Regression
        fc1 = tf.matmul(lstm_outputs[-1], wd1) + bd1
        #y = tf.nn.softmax(fc1)

        return fc1

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

