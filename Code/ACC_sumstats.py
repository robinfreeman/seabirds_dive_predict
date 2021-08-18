#!/usr/bin/env python3

""" Creates data set from a rolling window of data points (ACC/IMM) with corresponding class labels (dive/non-dive) for
training an artificial neural network to predict diving behaviour. """

__author__ = 'Luke Swaby (lds20@ic.ac.uk)'
__version__ = '0.0.1'

## Imports ##
import numpy as np
import core  # project module
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse

## Functions ##

def parse_arguments():
    """Function to parse args from command line
    """
    parser = argparse.ArgumentParser(
    description="Script for generating large data set for deep learning "
                "from rolling window of acceleration values..")

    parser.add_argument('-w', dest='window', type=int, default=2, help='Rolling time window width (rows).')

    args = parser.parse_args()

    print(f'\nPARAMS USED:\n'
          f'window:\t{args.window}s\n')

    return args.window


def main(wdw):
    """
    Creates a data set by taking a rolling window of 250 rows of the input
    arr and horizontally arranging x, y, and z values followed by a binary
    value indicating whether or not a dive has occured in that window.

    Arguments:
     - indir: (str) path to directory containing raw data
     - dtype: (str) predictor (ACC/IMM)
     - outpth: (str) path to directory to write data sets
     - wdw: (int) Rolling time window width (s).
     - threshold: (float) Depth threshold for identifying dives (m)
     - res: (int) resolution of predictor
     - reduce: (bool) reduce data set to balance class distribution?

    Output:
     - if reduce: pandas dataframe containing rolling windows of data points with corresponding class labels
       else: npy stack containing full rolling window data set
    """
    ######## CREATE DATA ##########
    no = int(wdw/(25*60))
    acc = pd.read_csv(f'../Data/BIOT_DGBP/ACCTest_{no}_mins.csv')

    dlist = []

    # undersample and concat again
    print('Reducing...')
    for bird in set(acc.TagID):
        df = acc[acc.TagID == bird]  # subset

        ndive = np.count_nonzero(df.Dive)
        dives = df[df.Dive == 1]
        nondives = df[df.Dive == 0]
        nondivesubset = nondives.sample(n=ndive)

        conc = pd.concat([dives, nondivesubset])
        conc = conc.sample(frac=1)  # shuffle

        dlist.append(conc)

    data = pd.concat(dlist)
    data = data.sample(frac=1)  # shuffle
    data.ix = data.ix.astype(int)

    data = data.drop(columns=['Max.ACC', 'SD.ACC'])

    # Scale
    scaler = MinMaxScaler()  # create scaler
    normalized = scaler.fit_transform(data.iloc[:, 1:3].to_numpy())  # fit and transform in one step
    data.iloc[:, 1:3] = normalized  # balanced, shuffled, and scaled

    metrics = ['Accuracy', 'AUC', 'Precision', 'Sensitivity', 'Specificity',
               'TruePos', 'FalsePos', 'FalseNeg', 'TrueNeg']

    outdf = pd.DataFrame(columns=metrics)

    # Cross validate
    for bird in set(data.TagID):

        train = data[data.TagID != bird]
        test = data[data.TagID == bird]

        X_train = train.drop(columns=['ix', 'Max.depth', 'Dive', 'TagID']).to_numpy()
        y_train = train.Dive.to_numpy()

        X_test = test.drop(columns=['ix', 'Max.depth', 'Dive', 'TagID']).to_numpy()
        y_test = test.Dive.to_numpy()

        # build model
        model = core.build_binary_classifier(in_shape=X_train[0].shape)

        # train model
        model.fit(X_train, y_train, epochs=50, batch_size=64)

        # test model
        m = model.evaluate(X_test, y_test)

        # Calculate extra stats
        conf_matrix = np.array(m[-4:])
        specificity = conf_matrix[-1] / (conf_matrix[1] + conf_matrix[-1])

        outdf.loc[bird] = [*m[1:5], specificity, *conf_matrix]

    outdf.index.name = 'TagID'
    outdf.to_csv(f'../Results_ACC_sumstats/meanACCandSAD/{no}_mins_xval.csv', header=True, index=True)

    return 0

if __name__ == '__main__':
    main(parse_arguments())