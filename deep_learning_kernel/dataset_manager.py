import numpy as np
import math

class Dataset_Manager():
    def __init__(self, dataset_split, TRAIN_VAL_SPLIT, sequence_length, batch_size):
        train_samples = int(round(dataset_split*TRAIN_VAL_SPLIT))
        validation_samples = int(dataset_split-train_samples)
        print("The dataset contains {} train samples and {} validation samples which is a {} ratio".format(
            train_samples, validation_samples, TRAIN_VAL_SPLIT))

        index = np.arange(dataset_split)
        np.random.shuffle(index)

        self.validation_split = index[validation_samples:]
        self.train_split = index[:train_samples]
        self.seq_length = sequence_length

        self.total_batches_train = math.ceil(len(self.train_split)/batch_size)
        self.total_batches_validation = math.ceil(len(self.validation_split)/batch_size)
        self.batch_size = batch_size

    def generate_batch(self):

        np.random.shuffle(self.train_split)

        np.random.shuffle(self.validation_split)
        print(":)")

    def get_next_batch(self, data, train, step):
        print(":)")
        minibatch = []
        minilabels = []
        if train:
            for i in range(self.batch_size):
                start_index = self.train_split[(self.batch_size*step)+i]*self.seq_lentgh
                train_batch = data[start_index:(start_index+self.seq_length), 0]
                train_y = data[start_index+self.seq_length-1, 1]
                minibatch.append(train_batch)
                minilabels.append(train_y)
        else:
            for i in range(self.batch_size):
                start_index = self.validation_split[(self.batch_size*step)+i]*self.seq_lentgh
                val_batch = data[start_index:(start_index+self.seq_length), 0]
                val_y = data[start_index+self.seq_length-1, 1]
                minibatch.append(val_batch)
                minilabels.append(val_y)
        return minibatch, minilabels
