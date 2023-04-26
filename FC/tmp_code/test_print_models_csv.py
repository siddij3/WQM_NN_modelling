
# ## Importing Data


# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
from pandas import DataFrame
import functions

import sql_manager
from sql_manager import sql_to_pandas


import numpy as np

# # Neural Network Creation and Selection Process
def Pearson(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True,
    )

    tmp_numerator, tmp_denominator_real,  tmp_denominator_pred = 0, 0,0

    i = 0


    while i < len(y_pred):
        tmp_numerator += (y_true[i] - sum(y_true)/len(y_true))* (y_pred[i] - sum(y_pred)/len(y_pred))

        tmp_denominator_real += (y_true[i] - sum(y_true)/len(y_true))**2
        tmp_denominator_pred += (y_pred[i] - sum(y_pred)/len(y_pred))**2
        i += 1


    R = tmp_numerator / (math.sqrt(tmp_denominator_pred) * math.sqrt(tmp_denominator_real))
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)

    return MAE, MSE, R[0], y_pred.flatten()

# ## Functions for Isolating Parameters
def loop_folds(neuralNets, _predictions, R, mae, mse, k_folds, features, labels, param1, param2, inner_val, outer_val, batch, vbs, str_test):

    tmp_mae, tmp_mse, tmp_R = [None]*k_folds, [None]*k_folds, [None]*k_folds
    avg_predictions = [None]*k_folds


    for j, NN in enumerate(neuralNets):

        tmp_mae__, tmp_mse__, tmp_R__, tmp_predictions = Pearson(NN, features, labels, batch, vbs) 

        tmp_R[j] = tmp_R__
        tmp_mse[j] = tmp_mse__
        tmp_mae[j] = tmp_mae__

        #dict_title = f"Predicted NN {j} for {param1},{inner_val}; {param2},{outer_val} - {str_test}"
        #_predictions[dict_title] = tmp_predictions.tolist()

        avg_predictions[j] = tmp_predictions.tolist()
    
    dict_average = f"Averages for {param1},{inner_val}; {param2},{outer_val}"
    dict_title_real = f"Real for {param1},{inner_val}; {param2},{outer_val} - {str_test}"

    _predictions[dict_title_real] = labels.tolist()

    arr_avg_predictions = np.transpose(avg_predictions)
    _predictions[dict_average] = [np.mean(i) for i in arr_avg_predictions]

    R.append(sum(tmp_R)/len(tmp_R))
    mae.append(sum(tmp_mae)/len(tmp_mae))
    mse.append(sum(tmp_mse)/len(tmp_mse))

    return _predictions, R, mae, mse

def isolateParam(optimal_NNs, data, table_name, engine, parameter, batch, verbose, str_test): 
    # Split the data labels with time
    unique_vals_inner = np.unique(data[parameter]) 

    list_of_feats = [None]*len(unique_vals_inner)
    list_of_labels = [None]*len(unique_vals_inner)
    
    for i, inner  in enumerate(unique_vals_inner):

        list_of_feats[i] = sql_manager.get_vals(table_name, 
                                                    parameter, inner, 
                                                    engine)
        
        list_of_labels[i] = list_of_feats[i].pop('Concentration')

        list_of_feats[i] = functions.get_dict(std_scaler.transform(list_of_feats[i].to_numpy()))

    tr_mae = []
    tr_mse = []
    tr_R = []
    _predictions = {}

    for i, time_vals in enumerate(list_of_feats):

        # print(f'{parameter1}: {unique_vals_inner[j]}', f'{parameter2}: {unique_vals_outer[j]}')

        _predictions, tr_R, tr_mae, tr_mse = loop_folds(optimal_NNs, _predictions, 
        tr_R, tr_mae,  tr_mse, 
        k_folds, 
        list_of_feats[int(i)], list_of_labels[int(i)],  
        parameter, "", 
        int(i), None, 
        batch, vbs, str_test)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{str_test} - {parameter} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)
    
    average_R = [i for i in tr_R]
    average_mae = [i for i in tr_mae]
    average_mse = [i for i in tr_mse]

    return average_R, average_mae, average_mse

def isolateTwoParam(optimal_NNs, data, table_name, engine, parameter1, parameter2, batch, vbs, str_test):

    unique_vals_inner = np.unique(data[parameter1]) 
    unique_vals_outer = np.unique(data[parameter2])

    list_of_feats = [None]*len(unique_vals_inner)
    list_of_labels = [None]*len(unique_vals_inner)

    for i, inner  in enumerate(unique_vals_inner):

        tmp_feats = [None]*len(unique_vals_outer)
        tmp_labels = [None]*len(unique_vals_outer)

        for j, outer in enumerate(unique_vals_outer):
            tmp_feats[j] = sql_manager.get_two_vals(engine, table_name, 
                                                     parameter1, inner, 
                                                      parameter2, outer)
            
            tmp_labels[j] = tmp_feats[j].pop('Concentration')

            tmp_feats[j] = functions.get_dict(std_scaler.transform(tmp_feats[j].to_numpy()))

        list_of_labels[i] = tmp_labels
        list_of_feats[i] = tmp_feats

    tr_mae = []
    tr_mse = []
    tr_R = []
    _predictions = {}
    for i, time_vals in enumerate(list_of_feats):
        tr_tmp_mae, tr_tmp_mse, tr_tmp_R = [], [], []

        for j, rsu_vals in enumerate(time_vals):
            

            # print(f'{parameter1}: {unique_vals_inner[j]}', f'{parameter2}: {unique_vals_outer[j]}')

            _predictions, tr_tmp_R, tr_tmp_mae, tr_tmp_mse  = loop_folds(optimal_NNs, _predictions,     
            tr_tmp_R, tr_tmp_mae,  tr_tmp_mse, 
            k_folds, 
            rsu_vals, list_of_labels[i][j],   
            parameter1, parameter2, 
            j, i, 
            batch, vbs, str_test)

        tr_mae.append(tr_tmp_mae)
        tr_mse.append(tr_tmp_mse)
        tr_R.append(tr_tmp_R)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{str_test} - {parameter1} and {parameter2} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)

 
    averages_mae = [[j for j in i] for i in tr_mae] 
    averages_R = [[j for j in i] for i in tr_R] 
    averages_mse = [[j for j in i] for i in tr_mse] 

    return averages_R, averages_mae, averages_mse


if __name__ == '__main__':

    ## DATA IMPORTING AND HANDLING
    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect()

    k_folds = 4
    sum_nodes = 38 #32
    num_epochs = 400 #500
    verbose = 0


    path = ".\\gold_fc_dev\\"
    nnmodel = [19,19]
    optimal_NNs = [ load_model(f'{path}Model {nnmodel} 0'), load_model(f'{path}Model {nnmodel} 1'),  load_model(f'{path}Model {nnmodel} 2'), load_model(f'{path}Model {nnmodel} 3') ]


    start_index= 0
    end_index = 3
    vbs = 0
    start_time = 1
    param_batches = 10


    str_reg = "All"
    table_name_print_data = "gold_fc"

    dataset = sql_to_pandas(table_name, engine)
    print_data = dataset

    #['mean_', 'scale_', 'var_']
    std_scaler = StandardScaler()
    std_ = sql_to_pandas("std_params_gold", engine)
 
    train_features = sql_to_pandas("gold_fc_train", engine)
    train_labels = train_features.pop('Concentration')
    train_features = functions.get_dict(std_scaler.fit_transform(train_features.to_numpy()))


    str_time = 'Time'
    str_rinse = 'Rinse'
    str_temp =  'Temperature'
    str_ph = 'pH Elapsed'


    
    R_time , mae_averages_time,  mse_averages_time  = isolateParam(optimal_NNs, print_data, table_name_print_data, engine, str_time, param_batches, vbs, str_reg)
    R_rinse , mae_averages_rinse,  mse_averages_rinse  = isolateParam(optimal_NNs, print_data, table_name_print_data, engine, str_rinse, param_batches, vbs, str_reg)

    dict_time = {
        "Time:  R"    : [i for i in R_time] , 
        "Time:  MAE"  : [i for i in mae_averages_time] , 
        "Time:  MSE"  : [i for i in mse_averages_time] , 
        
        "Rinse:  R"  : [i for i in R_rinse] , 
        "Rinse:  MAE"  : [i for i in mae_averages_rinse] , 
        "Rinse:  MSE"  : [i for i in mse_averages_rinse] , 

        }

    # print("Isolating Spin Coating and Time")
    R_of_sct , mae_of_sct, mse_of_sct  = isolateTwoParam(optimal_NNs, print_data, table_name_print_data, engine, str_rinse, str_time, param_batches, vbs, str_reg)


    dict_sc = {
        # "SC: R"    : R_of_sc ,
        # "SC: MAE"  : mae_of_sc ,
        "Time": [i for i in range(0, 51)],

        "Time SC: 0;: R"    : [i for i in R_of_sct[0] ], 
        "Time SC: 1: R"    : [i for i in R_of_sct[1] ],     

        "Time SC: 0 : MAE"  : [i for i in mae_of_sct[0] ], 
        "Time SC: 1 : MAE"  : [i for i in mae_of_sct[1] ], 

        "Time SC: 0 : MSE"  : [i for i in mse_of_sct[0] ], 
        "Time SC: 1 : MSE"  : [i for i in mse_of_sct[1] ], 

    }

    # #  Isolating Spin Coating and Time
    # #  Isolating Time
    # # print("Isolating Time")



    # # #  Isolating Increasing
    # # print("Isolating Increasing")
    # R_of_increasing , mae_of_increasing  = isolateParam(optimal_NNs , print_data, str_increasing, param_batches, vbs, str_reg )


    # # #  Isolating Repeat Sensor Use
    # # #print("Isolating Repeat Sensor Use")

    # # # # Printing to CSV

    dict_all =  dict_time | dict_sc # | dict_inc  | dict_abc  |     dict_repeat           # |  | dict_daysElapsed_time # 
    dict_all = DataFrame({ key:pd.Series(value) for key, value in dict_all.items() })
    dict_all.to_csv(f'{str_reg} - Final - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv')
