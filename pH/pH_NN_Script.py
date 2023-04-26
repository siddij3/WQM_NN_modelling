
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from enum import unique
from pandas import read_csv
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
#from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error

import math
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
from pandas import DataFrame
from multiprocessing import Process
from multiprocessing import Manager


import tensorflow as tf
import numpy as np




def get_dict(tt_feats):
    dict = {
        'Time':tt_feats[:, 0],
        'Voltage':tt_feats[:, 1], 
        'Days Elapsed':tt_feats[:, 2],
        'Sensor Cycle':tt_feats[:, 3], 
        'Start':tt_feats[:, 4], 
        'Increasing':tt_feats[:, 5], 
        'Decreasing':tt_feats[:, 6], 
        'Temperature': tt_feats[:, 7],
        'Repeat Use':tt_feats[:, 8], 
        'Sensor Number': tt_feats[:, 9],
        # 'Measurement Number':tt_feats[:, 10],
        'Measurement Continuous':tt_feats[:, 11]
    }
    return DataFrame(dict)

def importData(data, scaler):

    train_dataset = data.sample(frac=0.8, random_state=5096)
    test_dataset = data.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('pH')
    test_labels = test_features.pop('pH')

    train_features = get_dict(scaler.fit_transform(train_features.to_numpy()))
    test_features = get_dict(scaler.fit_transform(test_features.to_numpy()))

    #For later use
    data_labels = data.pop('pH')

    return data, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, 

# # Neural Network Creation and Selection Process
# 
def build_model(n1, n2):
  #Experiment with different models, thicknesses, layers, activation functions; Don't limit to only 10 nodes; Measure up to 64 nodes in 2 layers
  
    model = Sequential([
    layers.Dense(n1, activation=tf.nn.relu, input_shape=[11]),
    layers.Dense(n2, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
    #early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) 

    return model #, early_stop

def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, n1, n2, return_dict):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    model = build_model(n1, n2) #, early_stop = build_model(n1, n2)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, validation_split=0.3, verbose=verbose #, callbacks=early_stop
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    return_dict[i] = (model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_R)

def KCrossLoad(i, model, features, labels, num_val_samples, epochs, batch, verbose, n1, n2, return_dict):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, validation_split=0.3, verbose=verbose #, callbacks=early_stop
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    return_dict[i] = (model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_R)



def Pearson(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True,
    )


    tmp_numerator_real, tmp_numerator_pred, tmp_denominator_real,  tmp_denominator_pred = 0, 0, 0, 0

    i = 0
    while i < len(y_pred):


        tmp_numerator_real += y_true[i] - sum(y_true)/len(y_true)
        tmp_numerator_pred += y_pred[i] - sum(y_pred)/len(y_pred)

        tmp_denominator_real += (y_true[i] - sum(y_true)/len(y_true))**2
        tmp_denominator_pred += (y_pred[i] - sum(y_pred)/len(y_pred))**2

        i += 1


    if ((tmp_numerator_real == 0.0) & (tmp_denominator_real == 0.0)):
        tmp_numerator_real = 1
        tmp_denominator_real = 1


    R = (tmp_numerator_real*tmp_numerator_pred) / (math.sqrt(tmp_denominator_pred) * math.sqrt(tmp_denominator_real))

    return R[0], y_pred.flatten()

def MAE(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True,
    )

    MAE = mean_absolute_error(y_true, y_pred)
    return MAE

def scaleDataset(scaleData):
    scaleData = std_scaler.fit_transform(scaleData.to_numpy())
    return DataFrame(get_dict(scaleData))

def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# ## Functions for Isolating Parameters
def loop_folds(neuralNets, _predictions, R, mae, k_folds, features, labels, param1, param2, inner_val, outer_val, batch, vbs, str_test):

    tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
    avg_predictions = [None]*k_folds

    for j, NN in enumerate(neuralNets):
        test_mae = MAE(NN, features, labels, batch, vbs)

        tmp, tmp_predictions = Pearson(NN, features, labels, batch, vbs) 
        tmp_R[j] = tmp
        tmp_mae[j] = test_mae

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

    return _predictions, R, mae

def isolateParam(optimal_NNs, data, parameter, batch, verbose, str_test): 
    # Split the data labels with time

    unique_vals = np.unique(data[parameter]) #Repeat Sensor Use
    param_index= [np.where(data[parameter].to_numpy()  == i)[0] for i in unique_vals]

    scaled_features = scaleDataset(all_features.copy())
    #The full features of the data points that use certain time values

    param_features =  [scaled_features.iloc[param_index[int(i)]] for i,val in enumerate(unique_vals)]

    param_labels = [data_labels.to_numpy()[param_index[int(i)]] for i,val in enumerate(unique_vals)]


    mae, R = [], []
    _predictions = {}

    for i, val in enumerate(unique_vals):
        print(f'{parameter}: {val}')

        _predictions, R, mae = loop_folds(optimal_NNs, _predictions, 
        R, mae, 
        k_folds, 
        param_features[int(i)], param_labels[int(i)],   
        parameter, "", 
        val, None, 
        batch, verbose, str_test)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{str_test} - {parameter} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)
    
    average_R = [i for i in R]
    average_mae = [i for i in mae]

    return average_R, average_mae
 
def isolateTwoParam(optimal_NNs, data, parameter1, parameter2, batch, vbs, str_test):

    unique_vals_inner = np.unique(data[parameter1]) #Repeat Sensor Use
    unique_vals_time = np.unique(data[parameter2]) #Times

    inner = [[x for _, x in data.groupby(data[parameter1] == j)  ][1] for j in unique_vals_inner]   
    time_use = [[[x.index.values for _, x in data.groupby(val[parameter2] == j)  ][1] for val in inner] for j in unique_vals_time] 
 
    scaled_features = scaleDataset(all_features.copy())
    
    feats = [[scaled_features.iloc[sc]  for sc in rsu] for rsu in time_use] 
    labels = [[data_labels.to_numpy()[sc]  for sc in rsu] for rsu in time_use]

    tr_mae = []
    tr_R = []
    _predictions = {}
    for i, time_vals in enumerate(feats):
        tr_tmp_mae, tr_tmp_R = [], []

        for j, rsu_vals in enumerate(time_vals):

            print(f'{parameter1}: {unique_vals_inner[j]}', f'{parameter2}: {unique_vals_time[i]}')
            
            _predictions, tr_tmp_R, tr_tmp_mae = loop_folds(optimal_NNs, _predictions, 
            tr_tmp_R, tr_tmp_mae, 
            k_folds, 
            rsu_vals, labels[i][j],   
            parameter1, parameter2, 
            j, i, 
            batch, vbs, str_test)


        tr_mae.append(tr_tmp_mae)
        tr_R.append(tr_tmp_R)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{str_test} {parameter1} and {parameter2} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)

 
    averages_mae = [[j for j in i] for i in tr_mae] 
    averages_R = [[j for j in i] for i in tr_R] 

    return averages_R, averages_mae

def daysElapsed(optimal_NNs, data, param1, param2,  batch, vbs): 
    # Split the data labels with spin coating first

    param3 = 'Time'
    unique_vals_sc = np.unique(data[param1]) #Spin Coating
    unique_vals_days = np.unique(data[param2]) #Days Elapsed
    unique_vals_time = np.unique(data[param3])

    days = [[x for _, x in data.groupby(data[param2] == j)  ][1] for j in unique_vals_days]
    time_days = [[[x for _, x in data.groupby(val[param3] == j)  ][1] for val in days] for j in unique_vals_time] 

    #[time][day elapsed][spin coated]    
    all_vals = [[[x.index.values for _, x in data.groupby(unique_days[i][param1] == 0)] for i,val in enumerate(unique_days)] for unique_days in time_days] 
    
    scaled_features = scaleDataset(all_features.copy())

    feats = [[[scaled_features.iloc[sc]  for sc in days] for days in times] for times in all_vals]
    labels = [[[data_labels.to_numpy()[sc]  for sc in days] for days in times] for times in all_vals]

    shared_mae, shared_R = [], []
    _predictions = {}


    for t, times in enumerate(feats):
        days_tmp_mae, days_tmp_R = [], []

        for d, days in enumerate(times):
            sc_tmp_mae, sc_tmp_R = [], []

            print(f'{param2}: {unique_vals_days[d]}', f'{param3}: {unique_vals_time[t]}')

            for sc, isSpin in enumerate(days):

                _predictions, sc_tmp_R, sc_tmp_mae = loop_folds(optimal_NNs, _predictions, 
                sc_tmp_R, sc_tmp_mae, 
                k_folds, 
                isSpin, labels[t][d][sc],   
                param1, param2, 
                d, sc, 
                batch, vbs, "All data")

            days_tmp_mae.append(sc_tmp_mae)
            days_tmp_R.append(sc_tmp_R)

        shared_mae.append(days_tmp_mae)
        shared_R.append(days_tmp_R)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{param1} - {param2} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)

    R_ = {}
    MAE_  = {}

    for d, days in enumerate(shared_R[0]):
        for sc, isSC in enumerate(shared_R[0][d]):
            tmp_time_R, tmp_time_MAE = [], []

            tmp_time_R = [ time[d][sc] for t, time in enumerate(shared_R)]
            tmp_time_MAE = [ shared_mae[t][d][sc]  for t, time in enumerate(shared_R)]

            MAE_title = f" {param2} {unique_vals_days[d]}: {param1} {sc}; MAE"
            R_title = f"   {param2} {unique_vals_days[d]}: {param1} {sc}; R"
            
            MAE_[MAE_title] = tmp_time_MAE
            R_[R_title] = tmp_time_R

    R_MAE = R_ | MAE_
    #MAE_ = DataFrame({ key:pd.Series(value) for key, value in R_MAE.items() })

    return R_MAE

if __name__ == '__main__':
    dataset = read_csv(r'.\\Data\\aggregated_data_pH.csv')
    dataset = shuffle(dataset)

    std_scaler = StandardScaler()

    # ## NEURAL NETWORK PARAMETERS
    # 
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = importData(dataset.copy(), std_scaler)
    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 17 #32

    num_epochs = 20 #400 #500
    batch_size = 500 #50
    verbose = 0

    print("\n")
    print("Number Folds: ", k_folds)
    print("Initial Layers: ", n1_start, n2_start)
    print("Total Nodes: ", sum_nodes)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("\n")

    best_architecture = [0,0]

    dict_lowest_MAE,dict_highest_R  = {}, {}
    best_networks, best_history = 0,0

    mae_best  = 10
    R_best  = 0

    # #### Where the Magic Happens
    for i in range(n2_start, sum_nodes):
        for j in range(n1_start, sum_nodes):
            if (i+j > sum_nodes):
                continue
            
            print("first hidden layer", j)
            print("second hidden layer", i)
            k_fold_mae, k_models, k_weights, k_mae_history, R_tmp = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds

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
                                                j, 
                                                i, return_dict))
                _futures[fold].start()   
                
            for job in _futures:
                job.join()

        # (model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_mse, test_R)
            for fold in range(k_folds):
                k_models[fold] = model_from_json(return_dict.values()[fold][0]) #model is a JSON file
                k_weights[fold] = return_dict.values()[fold][1]
                k_models[fold].set_weights(k_weights[fold])

                k_mae_history[fold] = return_dict.values()[fold][2]
                k_fold_mae[fold] = return_dict.values()[fold][3]

                R_tmp[fold] = return_dict.values()[fold][4]
                

        
            R_recent = sum(R_tmp)/len(R_tmp)
            mae_recent = sum(k_fold_mae)/len(k_fold_mae)


            dict_highest_R['R: {}, {}'.format(j, i)] = R_recent
            dict_lowest_MAE['MAE: {}, {}'.format(j, i)] = mae_recent

            if (mae_recent <= mae_best):
                mae_best = mae_recent
                best_networks = k_models
                best_architecture = [j,i]
                best_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
            
            print(mae_best, mae_recent, best_architecture)


    # 
    # Find the model with the lowest error
    optimal_NNs  = best_networks
    i = 0
    for model in optimal_NNs :
        model.save("Model {} number {}".format(best_architecture, i))
        print("Models saved")

        i +=1


    # Plotting Loss Transition
    smooth_mae_history = smooth_curve(best_history)

    #   _predictions =DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    dict_epochs = { 
        "Epochs" : range(1, len(best_history) + 1),
        "Lowest MAE": best_history,

        "Smoothed Epochs": range(1, len(smooth_mae_history) + 1),

        "Lowest MAE Smoothed": smooth_mae_history,
        "Smoothed Epochs": range(1, len(smooth_mae_history) + 1)

        }

    dict_epochs = dict_epochs | dict_highest_R | dict_lowest_MAE
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv('Evolution and Architecture - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)


    # 
    start_index= 0
    end_index = 3
    vbs = 1
    start_time = 1
    param_batches = 10

    str_reg = "All"
    str_test = "Test"

    str_time = 'Time'
    str_increasing = 'Increasing'
    str_start = 'Start'
    str_decreasing = 'Decreasing'
    str_repeat = 'Repeat Use'
    str_sens_num = 'Sensor Number'
    str_meas_cont = 'Measurement Continuous'

    def create_dict(str_param, loop_values, R_val, mae_val):

        str_r =  'R {} '.format(str_param)
        str_mae =  'MAE {}'.format(str_param)


        return {
        str_param: [i for i in loop_values],
        str_r    : R_val , 
        str_mae  : mae_val 
        }


    print("Isolating Time")
    R_time , mae_averages_time  = isolateParam(optimal_NNs , all_features, str_time, param_batches, vbs, str_reg )
    dict_time = create_dict(str_time, range(0, 51), R_time, mae_averages_time)

    print("Isolating increasing")
    R_increasing , mae_averages_increasing  = isolateParam(optimal_NNs , all_features, str_increasing, param_batches, vbs, str_reg )
    dict_increasing = create_dict(str_increasing, np.unique(all_features[str_increasing]), R_increasing, mae_averages_increasing)

    print("Isolating Decreasing")
    R_decreasing , mae_averages_decreasing  = isolateParam(optimal_NNs , all_features, str_decreasing, param_batches, vbs, str_reg )
    dict_decreasing = create_dict(str_decreasing, np.unique(all_features[str_decreasing]), R_decreasing, mae_averages_decreasing)

    print("Isolating Start")   #Get a histogram of this or something
    print(all_features)    
    R_start, mae_averages_start  = isolateParam(optimal_NNs , all_features, str_start, param_batches, vbs, str_reg )
    dict_start = create_dict(str_start, np.unique(all_features[str_start]), R_start, mae_averages_start)

    print("Isolating Repeat Use")
    R_repeat , mae_averages_repeat  = isolateParam(optimal_NNs , all_features, str_repeat, param_batches, vbs, str_reg )
    dict_repeat = create_dict(str_repeat, np.unique(all_features[str_repeat]), R_repeat, mae_averages_repeat)

    print("Measurement Continuous")
    R_meas_cont , mae_averages_meas_cont  = isolateParam(optimal_NNs , all_features, str_meas_cont, param_batches, 1, str_reg )
    dict_meas_cont = create_dict(str_meas_cont, np.unique(all_features[str_meas_cont]), R_meas_cont, mae_averages_meas_cont)

    print("Isolating Sensor Number")   # Plot this one
    R_sens_num, mae_averages_sens_num  = isolateParam(optimal_NNs , all_features, str_sens_num, param_batches, 1, str_reg )
    dict_sens_num = create_dict(str_sens_num, np.unique(all_features[str_sens_num]), R_sens_num, mae_averages_sens_num)

    # # Printing to CSV
    dict_all = dict_time  | dict_increasing | dict_decreasing | dict_start | dict_repeat | dict_meas_cont | dict_sens_num
    dict_all = DataFrame({ key:pd.Series(value) for key, value in dict_all.items() })
    dict_all.to_csv('Final - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

    print(best_architecture)


# Important Params: Time; Measurement continuous; Increasing and Decreasing on 1 plot; Repeat Use is consistent (explain the outliers)