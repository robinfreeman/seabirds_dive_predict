#!/usr/bin/env python3

import re
import glob
import core
import numpy as np
import pandas as pd
import multiprocessing
import dask.dataframe as dd
#from tensorflow.keras.callbacks import ModelCheckpoint
import argparse

def parse_arguments():
    """Function to parse args from command line
    """
    parser = argparse.ArgumentParser(
    description="Script for training multiple binary classifiers for multiple data sets — each corresponding to "
                "a different window size — for either immersion (IMM) or acceleration (ACC) data.")

    parser.add_argument('-i', dest='indir', type=str, default='../Data/Reduced/',
                        help='Path to directory containing in files.')
    parser.add_argument('-o', dest='outdir', type=str, default='../Results/',
                        help='Path to directory for out files.')
    parser.add_argument('-t', dest='dtype', type=str, choices=['ACC', 'IMM'], required=True,
                        help='Data sets to analyse (ACC/IMM).')
    parser.add_argument('-y', dest='ycol', type=str, default='Dive', help='Y column')
    parser.add_argument('-d', dest='drop', nargs='+', default=['BirdID', 'ix'],
                        help='Additional columns (except y column) to drop from training set')
    parser.add_argument('-e', dest='epochs', type=int, default=100,
                        help='Max no. of epochs to train each model.')

    # indir, outdir, dtype, ycol, drop, epochs = ('../Data/Reduced/', None, 'IMM', 'Dive', ['BirdID', 'ix'], 1)

    args = parser.parse_args()

    print(f'PARAMS USED:\n'
          f'indir:\t{args.indir}\n'
          f'outdir:\t{args.outdir}\n'
          f'dtype:\t{args.dtype}\n'
          f'y_col:\t{args.ycol}s\n'
          f'drop:\t{args.drop}\n'
          f'epochs:\t{args.epochs}')

    return args.indir, args.outdir, args.dtype, args.ycol, args.drop, args.epochs


def main(indir, outdir, dtype, ycol, drop, epochs):

    assert indir.endswith('/'), 'indir arg must end with a forward slash'
    assert outdir.endswith('/'), 'outdir arg must end with a forward slash'

    # DataFrame for out stats
    metrics = ['Accuracy', 'AUC', 'Precision', 'Sensitivity', 'Specificity',
               'TruePos (%)', 'FalsePos (%)', 'FalseNeg (%)', 'TrueNeg (%)']
    out_stats = pd.DataFrame(columns=metrics)

    # Parse files
    files = glob.glob(f'{indir}{dtype}*.csv')

    for f in files:

        print(f'\n---------- PROCESSING FILE: {f} ----------\n')

        # Extract model ID from filepath and load data
        wdw = re.search(fr"/{dtype}(\d+)_reduced", f).group(1)
        data = dd.read_csv(f)

        birds = set(data.BirdID)

        # Train a model for each bird withheld for testing
        with multiprocessing.Pool() as pool:
            m = pool.starmap(core.build_train_evaluate_dask,
                             [(data, bird, f'{outdir}{dtype}_{wdw}_Keras/{bird}_withheld.h5',
                               ycol, drop, epochs) for bird in birds])

        Xval_metrics = pd.DataFrame(m, columns=['BirdID', *metrics])

        # Convert confusion matrix stats to percentages
        conf_temp = Xval_metrics.iloc[:, -4:].to_numpy()
        Xval_metrics.iloc[:, -4:] = (conf_temp / conf_temp.sum(axis=1, keepdims=True)) * 100

        # Save metrics for current window size
        Xval_metrics.to_csv(f'{outdir}{dtype}_{wdw}_xval_metrics_keras.csv', header=True, index=False)

        # Aggeragate stats
        mean_stats = Xval_metrics.iloc[:, 1:6].mean(axis=0)
        conf_total = conf_temp.sum(axis=0)
        conf_total = (conf_total / conf_total.sum()) * 100

        out_stats.loc[wdw] = [*mean_stats, *conf_total]

        # Generate predictions
        with multiprocessing.Pool() as pool:
            preds = pool.starmap(core.predict_dives, [(f'{outdir}{dtype}_{wdw}_Keras/{bird}_withheld.h5',
                                                              data[data.BirdID == bird].compute(),
                                                              ycol, drop, True) for bird in birds])

        predictions = pd.concat(preds)
        predictions.to_csv(f'{outdir}{dtype}_{wdw}_xval_predictions.csv', header=True, index=False)

        # Train and save full model
        in_shape = np.zeros(len(set(data.columns) - {ycol, *drop})).shape  # input layer shape for classifier
        full_model = core.build_binary_classifier(in_shape=in_shape)
        full_model = core.train_classifier_dask(full_model, data, ycol=ycol, drop=drop, epochs=epochs)
        full_model.save(f'{outdir}{dtype}_{wdw}_Keras/full_model.h5')

    out_stats = out_stats.sort_index(ascending=True, axis=0)
    out_stats.index.name = 'Window Size (s)'
    out_stats.to_csv(f'../Results/{dtype}_WindowComp_XVal_Metrics_Keras.csv', header=True, index=True)

if __name__ == '__main__':
    main(*parse_arguments())