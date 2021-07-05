#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import random
import numpy as np
import dask.array as da
import dask.dataframe as dd
from tensorflow import keras


## Functions ##

def check_for_dive(arr, thrshold=0.03):
    """
    Determines whether a dive occurred in given window of depth values.

    Arguments:
     - arr: dask array of depth values
     - threshold: Depth threshold to determine dives

    Output:
     - Binary integer indicating whether or not a dive has occurred
    """
    return int((arr[~da.isnan(arr)] > thrshold).any())


def reduce_dset(data):
    """
    Extracts a balanced subset from a rolling window data set

    Arguments:
     - data: dask array containing rolling window data

    Output:
     - Shuffled, reduced pandas dataframe (roughly 50/50 dives/non-dives)
    """
    # Sample all dives
    dive_ix = da.where(data[:, -1] == 1)[0].compute()
    pos = data[dive_ix]
    n_dive = len(dive_ix)

    # Sample non-dives
    no_dive_ix = np.setdiff1d(np.arange(data.shape[0]), dive_ix)
    n_no_dive = random.randint(n_dive, round(n_dive * 1.1))
    neg = data[da.random.choice(no_dive_ix, n_no_dive, replace=False)]

    # Stack, shuffle and compute
    data_add = dd.from_dask_array(da.vstack((pos, neg))).compute()
    data_add = data_add.sample(frac=1)
    data_add.columns = [*data_add.columns[:-1], 'Dive']  # rename last col
    data_add['Dive'] = data_add['Dive'].astype(int)

    return data_add


def rolling_acceleration_window(arr, wdw, threshold, res=25):
    """
    Creates a large data set from a rolling window of tri-axial acceleration records and depth
    records for training a neural net.

    Arguments:
     - arr: 4-column dask array containing X, Y, Z, and Depth records (in that order)
     - wdw: int specifying window size in seconds
     - threshold: Depth threshold past which a dive can be said to have occurred
     - res: ACC data resolution (Hz)

    Output:
     - Rolling window dask array with each row consisting of flattened X, Y, Z vector followed by binary
       int indicating whether ot not a dive has occurred within that window
    """
    # assert wdw % res == 0, f'Window size must be divisible by {res} for {res}Hz ACC data'

    wdw *= res  # expand resolution to no. of rows

    x = da.lib.stride_tricks.sliding_window_view(arr[:, 0], wdw)
    y = da.lib.stride_tricks.sliding_window_view(arr[:, 1], wdw)
    z = da.lib.stride_tricks.sliding_window_view(arr[:, 2], wdw)
    depth = da.lib.stride_tricks.sliding_window_view(arr[:, 3], wdw)

    d = da.apply_along_axis(check_for_dive, 1, depth, threshold)
    train_data = da.hstack((x, y, z, d.reshape((d.shape[0], 1))))

    train_data.compute_chunk_sizes()

    return train_data


def rolling_immersion_window(arr, wdw, threshold, res=6):
    """
    Creates a large data set from a rolling window of immersion (wet/dry) and depth records for training a neural net.

    Arguments:
     - arr: 2-column dask array containing Immersion and Depth records (in that order)
     - wdw: int specifying window size in seconds
     - threshold: Depth threshold past which a dive can be said to have occurred
     - res: immersion data resolution (no. of seconds between each record)

    Output:
     - Rolling window dask array with each row consisting of immersion vector followed by binary int indicating whether
       or not a dive has occurred within that window.
    """
    assert wdw % res == 0, f'Window size must be divisible by {res}'
    if arr.dtype != float:
        arr = arr.astype(float)

    imm = da.lib.stride_tricks.sliding_window_view(arr[:, 0], wdw)
    imm = da.apply_along_axis(lambda a: a[~da.isnan(a)], 1, imm)

    depth = da.lib.stride_tricks.sliding_window_view(arr[:, 1], wdw)
    d = da.apply_along_axis(check_for_dive, 1, depth, threshold)

    train_data = da.hstack((imm, d.reshape((d.shape[0], 1))))

    train_data.compute_chunk_sizes()

    return train_data


def build_binary_classifier(in_shape, l1_units=200, l2_units=200, dropout=0.2):
    """
    Builds a 2-hidden-layer neural network with dropout for binary classification using Keras.

    Arguments:
     - in_shape: (tuple) shape of input layer
     - l1_units: (int) no. of units in layer 1
     - l2_units: (int) no. of units in layer 2
     - dropout: (float) dropout value for hidden layers

    Output:
     - Compiled Tensorflow binary classification model
    """
    # Build model
    model = keras.models.Sequential([
        keras.layers.Dense(units=l1_units, input_shape=in_shape, activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=l2_units, activation='relu'),
        keras.layers.Dropout(rate=dropout),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall',
                           'TruePositives', 'FalsePositives',
                           'FalseNegatives', 'TrueNegatives']
                  )
    return model
