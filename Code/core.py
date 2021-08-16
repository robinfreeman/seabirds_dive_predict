#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##

import random
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
#from tensorflow import keras


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
    data_add.columns = ['ix', *data_add.columns[1:-1], 'Dive']  # rename first and last cols

    # Change dtypes
    data_add.Dive = data_add.Dive.astype(int)
    data_add.ix = data_add.ix.astype(int)

    #data_add.datetime = pd.to_datetime(data_add.datetime)

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
    ix = arr[:-(wdw - 1), 0].astype(int)  # extract ix col
    arr_tmp = arr[:, 1:].astype(float)

    x = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, 0], wdw)
    y = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, 1], wdw)
    z = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, 2], wdw)
    depth = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, 3], wdw)
    d = da.apply_along_axis(check_for_dive, 1, depth, threshold)

    # reshape column vectors
    shift = round(wdw / 2)
    ix = ix.reshape((-1, 1)) + shift  # shift each ix to centre of window
    d = d.reshape((-1, 1))

    train_data = da.hstack((ix, x, y, z, d))

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

    wdw = int(wdw / res)  # expand resolution to no. of rows
    ix = arr[:-(wdw - 1), 0].astype(int)  # extract ix col
    arr_tmp = arr[:, 1:].astype(float)

    # wet/dry
    imm = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, 0], wdw)
    #imm = da.apply_along_axis(lambda a: a[~da.isnan(a)], 1, imm).astype(int)

    # LUX
    #lux = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, 1], wdw)
    #lux = da.apply_along_axis(lambda a: max(a[~da.isnan(a)]), 1, lux).reshape((-1, 1))

    # Depth
    depth = da.lib.stride_tricks.sliding_window_view(arr_tmp[:, -1], wdw)
    d = da.apply_along_axis(check_for_dive, 1, depth, threshold)

    # reshape column vectors
    shift = round(wdw / 2) * res * 25  # to shift ix to centre of window (because 25 ix per second)
    ix = ix.reshape((-1, 1)) + shift
    d = d.reshape((-1, 1))

    #train_data = da.hstack((ix, imm, lux, d))
    train_data = da.hstack((ix, imm, d))

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
    from tensorflow import keras  # to enable multiple threads

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
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall',
                           'TruePositives', 'FalsePositives',
                           'FalseNegatives', 'TrueNegatives']
                  )
    return model


def train_classifier_dask(model, train_data, ycol='Dive', drop=['TagID', 'ix'], epochs=50):
    """
    Iteratively trains model on partitions of dask dataframe

    Arguments:
     - model: keras model object
     - train_data: (dask dataframe) training data
     - ycol: (str) name of class column
     - drop: (list) list of additional column names to drop in order to leave only feature columns
     - epochs: (int) number of epochs to train models for

    Output:
     - Trained model
    """
    #from tensorflow.keras.callbacks import EarlyStopping  # to enable multiple threads

    for i in range(train_data.npartitions):

        train_i = train_data.get_partition(i).compute()  # getting one partition
        X_train = train_i.drop(columns=drop + [ycol]).to_numpy()
        y_train = train_i[ycol].to_numpy()

        #es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=15, min_delta=.005)

        try:
            #model.fit(X_train, y_train, epochs=epochs, verbose=0, callbacks=[es])
            model.fit(X_train, y_train, epochs=epochs, verbose=0)
        except ValueError:
            continue

    return model


def build_train_evaluate_dask(data, bird, modelpath, ycol='Dive', drop=['TagID', 'ix'], epochs=100):
    """
    Build and trains a keras neural network to predict diving behaviour, then evaluates with data from a withheld bird.

    Arguments:
     - data: (dask dataframe) dataframe containing all features and corresponding classes (dive/non-dive) for all birds
       in the data set.
     - bird: (str) Tag ID of bird to withhold for testing
     - ycol: (str) name of class column
     - drop: (list) list of additional column names to drop in order to leave only feature columns
     - epochs: (int) number of epochs to train models for

    Output:
     - list of classification metrics: ['accuracy', 'AUC', 'Precision', 'Recall', 'Specificity', 'TruePositives',
       'FalsePositives', 'FalseNegatives', 'TrueNegatives']
    """
    #from tensorflow.keras.callbacks import EarlyStopping

    # Split data
    print(f"Withholding bird '{bird}'...")
    train = data[data.TagID != bird]
    test = data[data.TagID == bird].compute()

    X_test = test.drop(columns=drop + [ycol]).to_numpy()
    y_test = test[ycol].to_numpy()

    # TODO: split test into test/validation here and pass validation into train_calassifier to allow early_stopping?

    # Build and train
    model = build_binary_classifier(in_shape=X_test[0].shape)
    model = train_classifier_dask(model, train, ycol=ycol, drop=drop, epochs=epochs)

    # Save and evaluate
    model.save(modelpath)
    m = model.evaluate(X_test, y_test, verbose=0)

    # Calculate extra stats
    conf_matrix = np.array(m[-4:])
    specificity = conf_matrix[-1] / (conf_matrix[1] + conf_matrix[-1])

    return [bird, *m[1:5], specificity, *conf_matrix]


def predict_dives(modelpath, data, ycol='Dive', drop=['TagID', 'ix'], add_ID_col = True):
    """
    Build and trains a keras neural network to predict diving behaviour, then evaluates with data from a withheld bird.

    Arguments:
     - modelpath: (str) path to model for generating predictions
     - data: (dask dataframe) dataframe containing features to classify
     - ycol: (str) name of class column
     - drop: (list) list of additional column names to drop in order to leave only feature columns
     - add_ID_col: (bool) add Tag ID col to out CSV?

    Output:
     - pandas dataframe containing model predictions and corresponding indicies for georeferencing
    """
    # modelpath = '../Results/Reduced/Keras_ACC_XVal_Results/ACC_2_Keras/ch_gps03_S1_withheld.h5'

    from tensorflow import keras  # internal import to enable multiple threads
    #from sklearn.metrics import roc_curve

    # Split data
    model = keras.models.load_model(modelpath)
    X_test = data.drop(columns=drop + [ycol]).to_numpy()
    #y_test = data[y_field]

    # Calculate optimal threshold
    # TODO: THIS CAN'T BE USED FOR BIRDS FOR WHICH NO DIVE COLUMN IS AVAILABLE!, STICK WITH 0.5...
    #fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
    #gmeans = np.sqrt(tpr * (1 - fpr))  # calculate the g-mean for each threshold
    #ix = np.argmax(gmeans)
    #threshold = thresholds[ix]

    # Calculate predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    #model.predict_classes(X_test)

    if add_ID_col:
        out_df = pd.DataFrame(zip(data.TagID, data.ix, y_pred), columns=['TagID', 'ix', 'Prediction'])
    else:
        out_df = pd.DataFrame(zip(data.ix, y_pred), columns=['ix', 'Prediction'])

    return out_df


