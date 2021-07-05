#!/usr/bin/env python3

from tensorflow import keras
import numpy as np
import pandas as pd
import re
import glob
import dask.dataframe as dd


def build_model(in_shape=(750,), l1_units=200, l2_units=200, dropout=0.2):
    """Builds a 2 layer neural network for binary classification with tf.keras.
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
                  metrics=['accuracy',
                           'AUC',
                           'Recall',
                           'TruePositives',
                           'FalsePositives',
                           'FalseNegatives',
                           'TrueNegatives'
                           ])
    return model

files = glob.glob('../Data/Reduced/ACC*.csv')
out_stats = pd.DataFrame(columns=['Accuracy', 'AUC', 'Sensitivity', 'Specificity',
                                  'TruePos (%)', 'FalsePos (%)', 'FalseNeg (%)', 'TrueNeg (%)'])

for f in files:

    # Load data
    data = dd.read_csv(f)

    metrics = pd.DataFrame(index=['Accuracy', 'AUC', 'Sensitivity', 'TruePos (%)',
                                  'FalsePos (%)', 'FalseNeg (%)', 'TrueNeg (%)'])

    conf_matrix = np.zeros((2, 2))

    # Cross validate
    for bird in set(data['BirdID']):

        # Split data
        train = data[data['BirdID'] != bird]
        test = data[data['BirdID'] == bird].compute()

        X_test = test.drop(columns=['Dive', 'BirdID']).to_numpy()
        y_test = test['Dive'].to_numpy()

        # Build model
        model = build_model(in_shape=X_test[0].shape)

        # Train model
        for i in range(train.npartitions):

            # getting one partition
            train_i = train.get_partition(i).compute()

            X_train = train_i.drop(columns=['Dive', 'BirdID']).to_numpy()
            y_train = train_i['Dive'].to_numpy()

            try:
                model.fit(X_train, y_train, epochs=50)
            except ValueError:
                continue

        # Evaluate model
        m = model.evaluate(X_test, y_test)

        # Save metrics
        conf_matrix += np.array(m[-4:]).reshape(2, 2)
        metrics[bird] = m[1:]

    metrics['mean'] = metrics.mean(axis=1)

    # Extract model ID from filepath
    wdw = re.search(r"/ACC(\d+)_reduced", f).group(1)

    out_stats.loc[wdw] = [metrics['mean']['Accuracy'], metrics['mean']['AUC'],
                          metrics['mean']['Sensitivity'], conf_matrix[1][1]/conf_matrix[1].sum(),
                          *((conf_matrix/conf_matrix.sum())*100).flatten()]

out_stats.sort_index(ascending=True, axis=0, inplace=True)
out_stats.to_csv('../Results/Keras_XVal_Metrics.csv', header=True, index=True)
