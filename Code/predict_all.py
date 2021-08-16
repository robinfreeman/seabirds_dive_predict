#!/usr/bin/env python3

""" Writes data set for deep learning model to predict dives using acceleration
data. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import glob
import dask.dataframe as dd
import re
from tensorflow import keras
import core
import numpy as np
import pandas as pd
import argparse


def parse_arguments():
    """Function to parse args from command line
    """
    parser = argparse.ArgumentParser(
    description="Script for training multiple binary classifiers for multiple data sets — each corresponding to "
                "a different window size — for either immersion (IMM) or acceleration (ACC) data.")

    parser.add_argument('-i', dest='indir', type=str, required=True,
                        help='Path to directory containing in files.')
    parser.add_argument('-r', dest='resdir', type=str, default='../Results/',
                        help='Path to directory for out files.')
    parser.add_argument('-d', dest='dtype', type=str, choices=['ACC', 'IMM'], required=True,
                        help='Data sets to analyse (ACC/IMM).')
    parser.add_argument('-w', dest='window', type=int, default=4,
                        help='Rolling window width.')
    parser.add_argument('-t', dest='threshold', type=float, default=0.1,
                        help='Depth threshold for identifying dives.')

    # indir, resdir, dtype, ycol, drop, threshold = ('../Data/GLS Data 2019 Jan DG RFB Short-term/matched/', '../Results_t0.1_NoLUX_RmNA/', 'IMM', 'Dive', ['TagID', 'ix'], 0.1)

    args = parser.parse_args()

    print(f'PARAMS USED:\n'
          f'indir:\t{args.indir}\n'
          f'outdir:\t{args.resdir}\n'
          f'dtype:\t{args.dtype}\n'
          f'window:\t{args.window}s\n'
          f'thold:\t{args.threshold}\n'
          )

    return args.indir, args.resdir, args.dtype, args.window, args.threshold




# TODO:
# 1. Load raw ACC/IMM data files
# 2. Recreate rolling window data

#wdw = 420
#indir = '../Data/GLS Data 2019 Jan DG RFB Short-term/matched/'
#dtype = 'IMM'
#resdir = '../Results_t0.1_NoLUX/'
#threshold = 0.1

#


def main(indir, resdir, dtype, wdw, threshold):

    files = glob.glob(f'{indir}{dtype}*.csv')

    for f in files:

        print(f'\nPROCESSING FILE: {f}...\n')

        # Create out file
        bird = re.search(fr"{dtype}_(\w+).csv", f).group(1)  # bird ID from path
        outpth = f'{resdir}{dtype}_{wdw}_{bird}_ALL_predictions.csv'
        pd.DataFrame(columns=['ix', 'TagID', 'Prediction', 'Dive']).to_csv(outpth, index=False, header=True)

        # 2. Recreate rolling window data
        if dtype == 'ACC':
            arr = dd.read_csv(f).to_dask_array(lengths=True)
            train_data = core.rolling_acceleration_window(arr, wdw, threshold, res=25)
        else:
            df = dd.read_csv(f, usecols=['ix', 'wet/dry', 'Depth_mod'], dtype={'Depth_mod': 'float64'})
            conv_nums = {'wet/dry': {'wet': 1, 'dry': 0}}
            arr = df[['ix', 'wet/dry', 'Depth_mod']].replace(conv_nums).to_dask_array(lengths=True)
            train_data = core.rolling_immersion_window(arr, wdw, threshold, res=6)

        # Convert to dataframe and rename columns
        train_data_df = dd.from_dask_array(train_data)
        train_data_df = train_data_df.rename(columns={0: "ix", (train_data.shape[1] - 1): "Dive"})

        # Load model
        modelpath = f'{resdir}{dtype}_{wdw}_Keras/{bird}_withheld.h5'
        model = keras.models.load_model(modelpath)

        # Predict and write to csv
        for i in range(train_data_df.npartitions):
            test_i = train_data_df.get_partition(i).compute()  # getting one partition

            ix = test_i.ix.to_numpy().astype(int).reshape((-1, 1))
            dives = test_i.Dive.to_numpy().astype(int).reshape((-1, 1))
            features = test_i.drop(columns=['ix', 'Dive']).to_numpy()

            # Calculate optimal threshold
            from sklearn.metrics import roc_curve, confusion_matrix
            # TODO: THIS CAN'T BE USED FOR BIRDS FOR WHICH NO DIVE COLUMN IS AVAILABLE!, STICK WITH 0.5...
            # fpr, tpr, thresholds = roc_curve(dives, model.predict(features))
            # gmeans = np.sqrt(tpr * (1 - fpr))  # calculate the g-mean for each threshold
            # idx = np.argmax(gmeans)
            # th = thresholds[idx]
            # y_pred = (model.predict(features) > th).astype(int).reshape((-1, 1))

            #features = features[:, :-1]  # drops light data TODO: hash this out when adding light data back

            y_pred = (model.predict(features) > 0.5).astype(int).reshape((-1, 1))

            new_chunk = np.hstack((ix, y_pred, dives))
            new_chunk = pd.DataFrame(new_chunk, columns=['ix', 'Prediction', 'Dive'])
            new_chunk['TagID'] = bird
            new_chunk = new_chunk[['ix', 'TagID', 'Prediction', 'Dive']]

            new_chunk.to_csv(outpth, mode='a', header=False, index=False)

"""
def main(indir, resdir, dtype, wdw, threshold):

    # Parse files
    files = glob.glob(f'{indir}{dtype}*.csv')

    for f in files:
        wdw = int(re.search(fr"/{dtype}(\d+)_reduced", f).group(1)) # get window
        # todo: swap hash
        birds = list(set(dd.read_csv(f, usecols=['TagID'])['TagID']))
        #birds = list(set(dd.read_csv(f, usecols=['BirdID'])['BirdID']))

        outpth = f'{resdir}{dtype}_{wdw}_xval_ALL_predictions.csv'
        pd.DataFrame(columns=['ix', 'TagID', 'Prediction', 'Dive']).to_csv(outpth, index=False, header=True)

        for bird in birds:

            modelpath = f'{resdir}{dtype}_{wdw}_Keras/{bird}_withheld.h5'
            r = re.compile(f'.*{bird}.*')

            if dtype == 'ACC':
                i_files = glob.glob('../Data/BIOT_DGBP/ACC*.csv')  # fetch original file
                f = list(filter(r.match, i_files))[0]
                arr = dd.read_csv(f).to_dask_array(lengths=True)
                train_data = core.rolling_acceleration_window(arr, wdw, threshold, res=25)
            else:
                i_files = glob.glob('../Data/GLS Data 2019 Jan DG RFB Short-term/matched/IMM*.csv')
                f = list(filter(r.match, i_files))[0]
                df = dd.read_csv(f, usecols=['ix', 'wet/dry', 'light(lux)', 'Depth_mod'],
                                 dtype={'Depth_mod': 'float64', 'light(lux)': 'float64'})
                conv_nums = {'wet/dry': {'wet': 1, 'dry': 0}}
                arr = df[['ix', 'wet/dry', 'light(lux)', 'Depth_mod']].replace(conv_nums).to_dask_array(lengths=True)
                train_data = core.rolling_immersion_window(arr, wdw, threshold, res=6)

            # Convert to dataframe and rename columns
            train_data_df = dd.from_dask_array(train_data)
            train_data_df = train_data_df.rename(columns={0: "ix", (train_data.shape[1] - 1): "Dive"})

            # Load model
            model = keras.models.load_model(modelpath)

            # Predict and write to csv
            for i in range(train_data_df.npartitions):

                test_i = train_data_df.get_partition(i).compute()  # getting one partition

                ix = test_i.ix.to_numpy().astype(int).reshape((-1, 1))
                dives = test_i.Dive.to_numpy().astype(int).reshape((-1, 1))
                features = test_i.drop(columns=['ix', 'Dive']).to_numpy()

                features = features[:, :-1]   # drops light data TODO: hash this out when adding light data back


                y_pred = (model.predict(features) > 0.5).astype(int).reshape((-1, 1))

                new_chunk = np.hstack((ix, y_pred, dives))
                new_chunk = pd.DataFrame(new_chunk, columns=['ix', 'Prediction', 'Dive'])
                new_chunk['TagID'] = bird
                new_chunk = new_chunk[['ix', 'TagID', 'Prediction', 'Dive']]

                new_chunk.to_csv(outpth, mode='a', header=False, index=False)

    return 0
"""

if __name__ == '__main__':
    main(*parse_arguments())