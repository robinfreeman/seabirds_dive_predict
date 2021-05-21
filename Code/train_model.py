#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import glob
import os

# TODO:
# How does batch size fit into this? Do I take batch size randomly from each file?
# i.e. another layer of complexity, or do I just treat each file as batch?

## Functions ##

# grab every npz file in 8/9 bird dirs for training and every npz file in the
# other for validation
birds = glob.glob('../Data/Acc/*/')

train = []
for bird in birds[:-1]:
    train += glob.glob(f'{bird}*.npz')

valid = glob.glob(f'{birds[-1]}*.npz')

partition = {'train': train, 'validate': valid}



"""
all_files_loc = "../Data/Acc/"
all_files = os.listdir(all_files_loc)

image_label_map = {
        "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
        for i in range(int(len(all_files)/2))}
partition = [item for item in all_files if "image_file" in item]
"""

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    """

    def __init__(self, file_list):
        """Initialization
        """
        self.file_list = file_list
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(len(self.file_list))

    def __getitem__(self, index):
        """Generate one batch of data
        """
        # Generate index of the batch
        #indexes = self.indexes[index:(index + 1)]

        # Find list of IDs
        #file_list_temp = [self.file_list[k] for k in indexes]

        file_temp = self.file_list[index]

        # Generate data
        X, y = self.__data_generation(file_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.file_list))

    def __data_generation(self, file_temp):
        """Generates data containing batch_size samples
        """
        dta = np.load(file_temp)
        dta = dta[dta.files[0]]

        # Store sample
        X = dta[:, :-1]

        # Store class
        y = dta[:, -1]

        return X, y

# ====================
# train set
# ====================

training_generator = DataGenerator(partition['train'])

hst = model.fit_generator(generator=training_generator,
                           epochs=200,
                           validation_data=validation_generator,
                           use_multiprocessing=True,
                           max_queue_size=32)
