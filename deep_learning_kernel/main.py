import tensorflow as tf
import numpy as np
import pandas as pd
import os
import model_lstm
import dataset_manager
from datetime import datetime

#Seeds
np.random.seed(1234)
tf.set_random_seed(1234)


# Variables
TRAIN_PATH = "/home/guillem/Downloads/LANL-Earthquake-Prediction/train.csv"
TARIN_VAL_SPLIT = 0.75
filewriter_path = "/home/guillem/LANLTrains"
checkpoint_path = "/home/guillem/LANLTrains"

# Local variables
seq_length = 150000
batch_size = 8
n_features = 1
starter_learning_rate = 0.0001
epochs = 50

print("Loading data")
train = pd.read_csv(TRAIN_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

print("Data with shape {} has the following type of data:".format(train.shape))
print(train.head())

dataset_split = np.floor(train.shape[0]/seq_length) #Number of samples used on training

dataset = dataset_manager.Dataset_Manager(dataset_split, TARIN_VAL_SPLIT, seq_length, batch_size)

# Model
x = tf.placeholder(tf.float32, [None, seq_length, n_features])
y = tf.placeholder(tf.float32, [None, 1])
is_training = tf.placeholder(tf.bool)

model = model_lstm.model(x, seq_length, [], is_training, 'LSTM')
score = model.y

# List of trainable variables
var_list = tf.trainable_variables()
print("----")

with tf.name_scope("cost_function"):
    print("Loading loss")
    loss = tf.losses.absolute_difference(y, score)
    tf.summary.scalar('Absolute error', loss)

with tf.name_scope("MAE"):
    eval_metrics_ops = tf.metrics.mean_absolute_error(y,score)
    tf.summary.scalar('MAE', eval_metrics_ops)

with tf.name_scope("train"):
    # add an optimiser
    print("Loading gradients")
    global_step = tf.Variable(0, trainable=False)
    print("Global step")
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100, 0.95, staircase=True)
    print("learning rate")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    print("Optimizer loading")
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    print("Grads")

for var in var_list:
        tf.summary.histogram(var.name, var)

for grad, var in grads:
    print(var)
    print(grad)
    print("-----")
    tf.summary.histogram(var.name + '/gradient', grad)




# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer_train = tf.summary.FileWriter(filewriter_path + '/train')
writer_test = tf.summary.FileWriter(filewriter_path + '/validation')

# Initialize a saver for store model checkpoints
saver = tf.train.Saver(save_relative_paths=True)

# setup the initialisation operator
init_op = tf.global_variables_initializer()

# TODO: This can be erased
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print("Total parameters:", total_parameters)

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

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
        for step in range(dataset.total_batches_train):
            gs = (epoch * dataset.total_batches_train) + step + 1

            if (step % 50 == 0):
                print("MSE: ", mse, "epoch ", epoch, " global step: ", gs, " training: ",
                      round(100 * float(step) / float(dataset.total_batches_train), 3), '%')

            batch_x = 0
            batch_y = 0

            s, _, mse = sess.run([merged_summary, optimizer, eval_metrics_ops],
                               feed_dict={x: batch_x, y: batch_y, global_step: gs, is_training: True})
            writer_train.add_summary(s, (epoch * num_batches) + step)

            for i in range(total_batches_test):
                if (i % 100 == 0):
                    print(" validating: ", round(100 * float(i) / float(total_batches_test), 3), '%')

                batch_test_x, batch_test_y = dataset.get_next_batch(train, True, i)

                s, mse_val, result = sess.run([merged_summary, eval_metrics_ops, score],
                                                 feed_dict={x: batch_test_x, y: batch_test_y, is_training: False})
                writer_test.add_summary(s, (epoch * dataset.total_batches_validation) + i + step)

                print("{} Saving checkpoint of model...".format(datetime.now()))
                # save checkpoint of the model for each epoch
                checkpoint_name = os.path.join(checkpoint_path,
                                                   'model_epoch' + str(checkpoints) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                checkpoints += 1

                print("{} Model checkpoint saved at {}".format(datetime.now(),checkpoint_name))
                print("{} MSE Accuracy Last = {:.4f}".format(datetime.now(), mse_val))
