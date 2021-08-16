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
from matplotlib import pyplot as plt


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

"""
def reduce_dset_dask(data, outfmt):

    assert outfmt in ['dask', 'pandas'], "outfmt must be one of 'dask'/'pandas'"

    # Sample all dives
    dive_ix = da.where(data[:, -1] == 1)[0].compute()
    pos = data[dive_ix]
    n_dive = len(dive_ix)

    # Sample non-dives
    no_dive_ix = np.setdiff1d(np.arange(data.shape[0]), dive_ix)
    n_no_dive = random.randint(n_dive, round(n_dive * 1.1))
    neg = data[da.random.choice(no_dive_ix, n_no_dive, replace=False)]

    ######
    stacked = da.vstack((pos, neg))
    data = stacked[:, 1:].astype(float).compute()
    datetime = pd.to_datetime(dd.from_dask_array(stacked[:, 0]).compute())

    data_add = dd.from_dask_array(stacked).compute()
    #######

    # Stack, shuffle and compute
    if compute:
        data_add = dd.from_dask_array(data_add).compute()
        data_add = data_add.sample(frac=1)
        data_add.columns = ['datetime', *data_add.columns[1:-1], 'Dive']  # rename first and last cols

        # Change dtypes
        data_add.Dive = data_add.Dive.astype(int)
        data_add.datetime = pd.to_datetime(data_add.datetime)

    else:
        data_add = dd.from_dask_array(da.vstack((pos, neg)))
        N = n_dive + n_no_dive
        data_add['index'] = dd.from_array(np.random.choice(N, N, replace=False))
        data_add = data_add[['index', *data_add.columns[:-1]]]  # rearrange cols

        data_add.columns = ['Index', 'datetime', *data_add.columns[2:-1], 'Dive']  # rename first and last cols

        # Change dtypes
        data_add.Dive = data_add.Dive.astype(int)
        data_add.datetime = dd.to_datetime(data_add.datetime)

        # change dtypes of all numeric columns to float
        cols = [str(x) for x in data_add.columns if isinstance(x, int)]
        data_add.columns = data_add.columns.map(str)
        for col in cols:
            data_add[col] = data_add[col].astype(float)

        #data_add.loc[:, numcols] = data_add.loc[:, numcols].astype(float)

        #data_add = data_add.set_index('index', sorted=True)

    return data_add
"""

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
    #ix = ix.reshape((ix.shape[0], 1))
    #d = d.reshape((d.shape[0], 1))
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

    # TODO: expand immersion instead to 6s res with max depth value in surrounding +-3s as labels? (saves need to prodcue large IMM dset) na.. more important tigs to spend time on
    # TODO: put ix at centre of each window

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


#def train_classifier(model, train_data, y_field='Dive', epochs=100, drop=['TagID'], model_checkpoint=None):
    """

    :param to_drop:
    :param train_data:
    :param model_checkpoint:
    :return:
    """
"""
    print(f'Training on {train_data.npartitions} partitions...')

    for i in range(train_data.npartitions):

        print(f'\n\nPARTITION {i}\n\n')

        train_i = train_data.get_partition(i).compute()  # getting one partition
        X_train = train_i.drop(columns=drop + [y_field]).to_numpy()
        y_train = train_i[y_field].to_numpy()

        es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=10, min_delta=.005)

        try:
            history = model.fit(X_train, y_train, epochs=epochs, verbose=0,
                                callbacks=[es, model_checkpoint] if model_checkpoint else [es])
        except ValueError:
            continue

        #############################################
        #from matplotlib import pyplot as plt
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        #plt.plot(history.history['loss'])
        #plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # summarize history for loss
        
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        #############################################
    

    return
"""
def train_classifier_dask(model, train_data, ycol='Dive', drop=['TagID', 'ix'], epochs=50):
    """

    :param data:
    :param ycol:
    :param drop:
    :param epochs:
    :return:
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

    :param to_drop:
    :param train_data:
    :param model_checkpoint:
    code_no: code name to go into model name to recognise what it is
    :return:
    """
    from tensorflow.keras.callbacks import EarlyStopping

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

    :param modelpath:
    :param data:
    :return: 3 column dataframe TagID datetime Predictions

    - assumes balanced dset

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


