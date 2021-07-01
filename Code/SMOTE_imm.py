#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import re
import glob
import dask_acc_write as dwa
import pandas
import dask.dataframe as dd
import dask.array as da
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

## Functions ##

def split_and_combine(arr1, arr2, N):
    """Splits 2 arrays by same number and zips them together
    """
    grps = []
    for x in zip(np.array_split(arr1, N), np.array_split(arr2, N)):
        grp = np.hstack(x)
        np.random.shuffle(grp)
        grps.append(grp)

    return grps


def build_model(in_shape=(750,), l1_units=200, l2_units=200, dropout=0.2,
                loss='binary_crossentropy'):
    """Builds a 2 layer neural network for binary classification with tf.keras.
    """
    model = keras.models.Sequential([
        keras.layers.Dense(units=l1_units, input_shape=in_shape,
                           activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=l2_units, activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss=loss,
                  metrics=['accuracy'])
    return model


def main(indir, outpth, wdw=250, threshold=0.03, reduce=True):
    """
    Creates numpy dataframe by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.
    """
    assert indir.endswith('/'), "indir arg must end with a '/'"
    if reduce:
        assert not outpth.endswith('/'), "Out path should be to file, not dir"

    # Grab list of reevant file paths
    files = glob.glob(f'{indir}*1.csv')

    for ix in len(files):
        #  leave file/bird at idx i out for testing

        test_file = files[ix]

        # Create model (could just be unpacked here)
        # TODO: add this function (rewatch vid geeze made)
        model = build_model(in_shape=train_data[0][:-1].shape)

        for no, file in enumerate([f for f in files if f != test_file], 1):
            # Train on all others

            print(f"PROCESSING FILE {no}: '{file}'...")

            # Create dset from rolling window
            print('\r\tCreating rolling window dset...')
            arr = dd.read_csv(file, usecols=['X', 'Y', 'Z', 'Depth_mod']).to_dask_array(
                lengths=True)
            train_data = dwa.rolling_acceleration_window(arr, wdw, threshold)
            #bird = re.search(r"/(\w+).csv", file).group(1)  # bird ID from path

            # Find idx of dives/non-dives
            dive_ix = da.where(train_data[:, -1] == 1)[0].compute()
            no_dive_ix = np.setdiff1d(np.arange(train_data.shape[0]), dive_ix)
            np.random.shuffle(dive_ix)
            np.random.shuffle(no_dive_ix)

            # split both into equally partitioned chunks and join
            N = len(train_data.chunks[0])*2  # twice no of chunks in dask array
            ix_grps = split_and_combine(dive_ix, no_dive_ix, N)

            # Iteratively pull into memory, augment with SMOTE, and train model
            for i in range(N):
                # Load and split data
                dta = train_data[ix_grps[i]].compute()  # LONG STEP
                x_train = dta[:, :-1]
                y_train = dta[:, -1]

                # Augment data
                smote = SMOTE(sampling_strategy='minority')
                x_sm, y_sm = smote.fit_resample(x_train, y_train)

                # Train model
                model.fit(x_sm, y_sm, epochs=100, batch_size=128)

            print(f"EVALUATING WITH FILE 9: '{test_file}'...")
            arr = dd.read_csv(test_file, usecols=['X', 'Y', 'Z', 'Depth_mod']).to_dask_array(
                lengths=True)
            test_data = dwa.rolling_acceleration_window(arr, wdw, threshold)

            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]
            mets = model.evaluate(x_test, y_test)

# TODO: evaluate some time later










if __name__ == '__main__':
    main(*dwa.parse_arguments())