#!/usr/bin/env python3

import h2o
from h2o.estimators import H2ODeepLearningEstimator
import glob
import re

h2o.init(min_mem_size='30G', max_mem_size="100G")

files = glob.glob('../Data/Reduced/ACC*.csv')

for f in files:

    # Load data
    data = h2o.import_file(f, header=1)
    data['Dive'] = data['Dive'].asfactor()
    data['BirdID'] = data['BirdID'].asfactor()
    
    # Extract model ID from filepath
    wdw = re.search(r"/ACC(\d+)_reduced", f).group(1)
    
    # Build, train, and cross-validate model
    dl_cross = H2ODeepLearningEstimator(model_id = 'ACC_window_' + wdw,
                                        distribution = "bernoulli",
                                        hidden = [200, 200],
                                        fold_column = 'BirdID',
                                        keep_cross_validation_models = True,
                                        keep_cross_validation_fold_assignment = True,
                                        keep_cross_validation_predictions = True,
                                        score_each_iteration = True,
                                        epochs = 50,
                                        train_samples_per_iteration = -1,
                                        activation = "RectifierWithDropout",
                                        #input_dropout_ratio = 0.2,
                                        hidden_dropout_ratios = [0.2, 0.2],
                                        single_node_mode = False,
                                        balance_classes = False,
                                        force_load_balance = False,
                                        seed = 23123,
                                        score_training_samples = 0,
                                        score_validation_samples = 0,
                                        stopping_rounds = 0)
    print('Training...')

    dl_cross.train(x = data.columns[1:-1],
                   y="Dive",
                   training_frame=data)
    
    # Save model
    print('Saving...')
    h2o.save_model(model=dl_cross, path="../Data/Reduced/H2O_ACC_XVal_Models/", force=True)
    
# Close session
h2o.shutdown()
