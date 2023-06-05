
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.utils import shuffle


import math
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
from pandas import DataFrame
from multiprocessing import Process
from multiprocessing import Manager

import sql_manager
from sql_manager import pandas_to_sql
import file_management

import shutil

from cloudinary.uploader import upload
from cloudinary.utils import cloudinary_url

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import logins


import tensorflow as tf
import numpy as np

from functions import Pearson
from functions import smooth_curve
from functions import get_model_folder_name

 
def build_model(input, n1, n2):
  #Experiment with different models, thicknesses, layers, activation functions; Don't limit to only 10 nodes; Measure up to 64 nodes in 2 layers
  
    model = Sequential([
    layers.Dense(n1, activation=tf.nn.relu, input_shape=[input]),
    layers.Dense(n2, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse']) #, run_eagerly=True)
    
    return model


def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, input_params, n1, n2, return_dict, folder_name):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    model = build_model(input_params, n1, n2) #, early_stop = build_model(n1, n2)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, 
        validation_split=0.3, verbose=verbose,
        workers=3,
        use_multiprocessing=True,
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    MAE, MSE, test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    model.save(f".\\{folder_name}\\Model [{n1}, {n2}] {i}")

    return_dict[i] = (history['val_mae'], history['val_mse'], test_mae, MSE, test_R)




if __name__ == '__main__':
## DATA IMPORTING AND HANDLING
    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect()

    # dataset = shuffle(sql_to_pandas(table_name, engine))
    dataset = shuffle(sql_manager.get_vals(table_name, "Rinse", 1, engine))

    print(dataset)
    std_scaler = StandardScaler()
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, std_scaler = file_management.importData(dataset.copy(), std_scaler)

    std_params = pd.DataFrame([std_scaler.mean_, std_scaler.scale_, std_scaler.var_], 
                       columns = train_features.keys())
    std_params['param_names'] = ['mean_', 'scale_', 'var_']

    table_name = 'std_params_gold'
    if (not sql_manager.check_tables(engine, table_name)):
        pandas_to_sql(table_name, std_params, engine)

    else:
        sql_manager.remove_table(table_name, engine)
        pandas_to_sql(table_name, std_params, engine)

    # ## PRINCIPAL COMPONENT ANALYSIS
    num_components =  6 #Minimum: Time, current, derivative

     # ## NEURAL NETWORK PARAMETERS
    # 
    k_folds = 3
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 15,15
    sum_nodes =  30

    num_epochs = 10 #400 #500
    batch_size = 16 #50
    verbose = 0

    folder_name = get_model_folder_name()

    print("\n")
    print("Number Folds: ", k_folds)
    print("Initial Layers: ", n1_start, n2_start)
    print("Total Nodes: ", sum_nodes)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("\n")

    best_architecture = [0,0]

    dict_lowest_MAE,dict_highest_R, dict_lowest_MSE  = {}, {}, {}
    best_networks, best_history = 0,0

    mae_best  = 10
    R_best  = 0

    # #### Where the Magic Happens
    #### Where the Magic Happens
    for i in range(n2_start, sum_nodes):
        for j in range(n1_start, sum_nodes):
            if (i+j > sum_nodes):
                continue
            
            print("first hidden layer", j)
            print("second hidden layer", i)
            k_fold_mae, k_fold_mse, k_mae_history, k_mse_history, R_tmp =  [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds

            _futures = [None]*k_folds
            manager = Manager()
            return_dict = manager.dict()

            for fold in range(k_folds):
                _futures[fold] = Process(target=KCrossValidation, 
                                              args=(  fold, 
                                                train_features, 
                                                train_labels, 
                                                num_val_samples, 
                                                num_epochs, 
                                                batch_size, 
                                                verbose, 
                                                num_components,
                                                j, 
                                                i, return_dict,
                                                folder_name))
                _futures[fold].start()   
                
            for job in _futures:
                job.join()

        # ( history['val_mae'], test_mae, test_mse, test_R)
        #     return_dict[i] = (history['val_mae'], test_mae, test_R, MSE)
            for fold in range(k_folds):
                k_mae_history[fold] = return_dict.values()[fold][0]
                k_mse_history[fold] = return_dict.values()[fold][1]

                k_fold_mae[fold] = return_dict.values()[fold][2]
                k_fold_mse[fold] = return_dict.values()[fold][3]
                R_tmp[fold] = return_dict.values()[fold][4]

        
            R_recent = sum(R_tmp)/len(R_tmp)
            mae_recent = sum(k_fold_mae)/len(k_fold_mae)
            mse_recent = sum(k_fold_mse)/len(k_fold_mse)


            dict_highest_R[f'R: {j}, {i}'] = R_recent
            dict_lowest_MAE[f'MAE: {j}, {i}'] = mae_recent

            dict_lowest_MSE[f'MSE: {j}, {i}'] = mse_recent

            if (mae_recent <= mae_best):
                mae_best = mae_recent
                best_architecture = [j,i]
                best_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
                best_history_mse = [ np.mean([x[z] for x in k_mse_history]) for z in range(num_epochs)]
            
            print(mae_best, mae_recent, best_architecture)

    # Delete all other models here instead
    optimal_NNs  = best_networks
    i = 0

    print(best_architecture)
    
    filepath = file_management.get_file_path()
    local_download_path = os.path.expanduser(filepath)
    print(local_download_path)

    for filename in os.listdir(local_download_path):    
        if str(best_architecture) in filename:
            continue;

        if f"Model" in filename:
            print(filename)
            shutil.rmtree(f".\\{filepath}\\{filename}", ignore_errors=False)

        i +=1

    # Plotting Loss Transition
    # TODO add MSE
    smooth_mae_history = smooth_curve(best_history)
    smooth_mse_history = smooth_curve(best_history_mse)

    #   _predictions =DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    dict_epochs = { 
        "Epochs" : range(1, len(best_history) + 1),
        "Lowest MAE": best_history,
        "Lowest MSE": best_history_mse,

        "Smoothed Epochs": range(1, len(smooth_mae_history) + 1),
        "Lowest MAE Smoothed": smooth_mae_history,
        "Lowest MSE Smoothed": smooth_mse_history,
        }
    

    zip_filename = folder_name
    public_name = folder_name

    shutil.make_archive(zip_filename, 'zip', f".\\{folder_name}")
    response = upload(f'{folder_name}.zip', 
                      public_id=f'{public_name}', 
                      resource_type='auto')

    print(response)
    


    
    dict_epochs = dict_epochs | dict_highest_R | dict_lowest_MAE
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv(f'Evolution and Architecture PCA - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv')
        
