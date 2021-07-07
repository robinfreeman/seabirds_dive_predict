#!/usr/bin/env python3

import re
import glob
import core
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tensorflow.keras.callbacks import ModelCheckpoint

files = glob.glob('../Data/Reduced/ACC*.csv')
out_stats = pd.DataFrame(columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity',
                                  'TruePos (%)', 'FalsePos (%)', 'FalseNeg (%)', 'TrueNeg (%)'])

for f in files:

    # Extract model ID from filepath
    wdw = re.search(r"/ACC(\d+)_reduced", f).group(1)

    # Load data
    data = dd.read_csv(f)

    # Initialise data storage objects
    metrics = pd.DataFrame(index=['Accuracy', 'AUC', 'Sensitivity', 'TruePos (%)',
                                  'FalsePos (%)', 'FalseNeg (%)', 'TrueNeg (%)'])
    conf_matrix = np.zeros((2, 2))

    # Save best model for each window size
    mc = ModelCheckpoint(f'../Results/ACC_{wdw}_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    # Cross validate
    for bird in set(data.BirdID):

        # Split data
        train = data[data.BirdID != bird]
        test = data[data.BirdID == bird].compute()

        X_test = test.drop(columns=['Dive', 'BirdID']).to_numpy()
        y_test = test.Dive.to_numpy()

        # Build model
        model = core.build_binary_classifier(in_shape=X_test[0].shape)

        # Train model
        core.train_classifier(model, train, model_checkpoint=mc)

        # Evaluate model
        m = model.evaluate(X_test, y_test)

        # Save metrics
        conf_matrix += np.array(m[-4:]).reshape(2, 2)
        metrics[bird] = m[1:]

    metrics['mean'] = metrics.mean(axis=1)

    out_stats.loc[wdw] = [metrics['mean']['Accuracy'], metrics['mean']['AUC'],
                          metrics['mean']['Sensitivity'], conf_matrix[1][1]/conf_matrix[1].sum(),
                          *((conf_matrix/conf_matrix.sum())*100).flatten()]

out_stats.sort_index(ascending=True, axis=0, inplace=True)
out_stats.to_csv('../Results/Keras_XVal_Metrics.csv', header=True, index=True)
