
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os


import functions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.preprocessing import StandardScaler
from keras.models import load_model

import pandas as pd
from pandas import DataFrame
import file_management
import sql_manager

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from sql_manager import sql_to_pandas

from sql_manager import pandas_to_sql_if_exists

import re
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    
    ## DATA IMPORTING AND HANDLING
    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect()

    # if the table doesn't exist, create it from the csv file, 
    # and send the file to 
    dataset = sql_to_pandas(table_name, engine)

    std_scaler = StandardScaler()
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, std_scaler = file_management.importData(dataset.copy(), std_scaler)
    
    std_params = pd.DataFrame([std_scaler.mean_, std_scaler.scale_, std_scaler.var_], 
                       columns = train_features.keys())
    std_params['param_names'] = ['mean_', 'scale_', 'var_']

    std_table_name = functions.get_std_params_table_name()
    pandas_to_sql_if_exists(std_table_name, std_params, engine, "replace")


    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 16 #32

    num_epochs = 10 #400 #500
    batch_size = 16 #50
    verbose = 0

    functions.load_from_cloud()
    path = file_management.get_file_path()
    local_download_path = os.path.expanduser(path)

    optimal_NNs = [None]*k_folds

    i = 0
    tmp = ""
    for filename in os.listdir(local_download_path):
        if "Model" in filename:
            optimal_NNs[i] = load_model(f"{path}\\{filename}")
            print(optimal_NNs[i].optimizer)
            tmp = re.findall(r'\d+', filename)
            i+=1

    best_architecture = tmp
    n1 = tmp[0]
    n2 = tmp[1]
    print(n1, n2)

    k_fold_mse, k_fold_mae, k_models, k_weights, k_mae_history, R_tmp, history = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds

    _futures = [None]*k_folds

    for fold in range(k_folds):
        i = fold

        reconstructed_model = optimal_NNs[fold]

        val_data = train_features[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_features[:i * num_val_samples], train_features[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_labels[:i * num_val_samples], train_labels[(i + 1) * num_val_samples:]],     axis=0)

        print('Training fold #', i)
        history = reconstructed_model.fit(
            partial_train_data, partial_train_targets,
            epochs=num_epochs, batch_size=batch_size, 
            validation_split=0.3, verbose=verbose,
            workers=3
        )

        history = DataFrame(history.history)

        test_loss, test_mae, test_mse = reconstructed_model.evaluate(val_data, val_targets, verbose=verbose)
        MAE, MSE, test_R, y =  functions.Pearson(reconstructed_model, val_data, val_targets.to_numpy(), batch_size, verbose )
        reconstructed_model.save(f".\\{path}\\Model [{n1}, {n2}] {i}")


    # ( history['val_mae'], test_mae, test_mse, test_R)
        k_mae_history[fold] = history['val_mae']
        k_fold_mae[fold] = test_mae
        k_fold_mse[fold] = test_mse

        R_tmp[fold] = test_R

        

    best_history_retrained = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]


    dict_epochs = { "Retrained Results": best_history_retrained }
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv(f'Evolution and Architecture Retrained - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv')

