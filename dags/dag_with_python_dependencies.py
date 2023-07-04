from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.append('/opt/airflow/dags/libs')

from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pandas as pd
from pandas import DataFrame

import cloudinary
import sqlalchemy as sa

import requests
import shutil

import numpy as np

import os
import re

default_args = {
    'owner': 'junaid',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)
}


def download_models():
    import libs.af_file_management as file_management
    import libs.af_functions as functions

    path = os.path.expanduser(file_management.get_file_path())

    for filename in os.listdir(path):
        if "Model " in filename and ".h5" in filename:
            print(os.listdir(path))
            os.remove(f'./{path}/{filename}')


    zip_public_id = f'{functions.get_model_folder_name()}.zip'

    download_url = cloudinary.utils.cloudinary_url(zip_public_id, 
                                                resource_type='raw')[0]
    
    response = requests.get(download_url, stream=True)

    save_path = f"./{zip_public_id}"

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    path = os.path.expanduser(file_management.get_file_path())
    shutil.unpack_archive(save_path, path)

    for filename in os.listdir(path):
        if "Model " in filename and ".h5" in filename:
            print(filename)

    print(response)
    print(os.listdir("dags/libs"))
    print(os.listdir("dags"))
    print(os.listdir())
    
   # print(os.listdir(os.path.expanduser(filepath)))


def retrain_models():
    import libs.af_file_management as file_management
    import libs.af_functions as functions
    import libs.af_sql_manager as sql_manager

    ## DATA IMPORTING AND HANDLING
    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect()

    # if the table doesn't exist, create it from the csv file, 
    # and send the file to 
    dataset = sql_manager.sql_to_pandas(table_name, engine)

    std_scaler = StandardScaler()

    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, std_scaler = file_management.importData(dataset.copy(), std_scaler)
  
    k_folds = functions.get_num_folds()
    optimal_NNs = [None]*k_folds

    num_val_samples = len(train_labels) // k_folds

    num_epochs = 10 #400 #500
    batch_size = 16 #50
    verbose = 0

    path = os.path.expanduser(file_management.get_file_path())

    i = 0
    tmp = ""
    for filename in os.listdir(path):
        if "Model " in filename and ".h5" in filename:
            optimal_NNs[i] = load_model(f"{path}/{filename}")
            tmp = re.findall(r'\d+', filename)
            i+=1

    print(optimal_NNs[0], optimal_NNs[1], optimal_NNs[2])

    n1 = tmp[0]
    n2 = tmp[1]
    print(n1, n2)

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
        
        print(MAE, MSE, test_R)

        save_file = f'{path}/ Model [{n1}, {n2}] {i}.h5'
        print(f'filename: {save_file}')
        reconstructed_model.save(save_file)

    for filename in os.listdir(path):
        if "Model " in filename and ".h5" in filename:
            print(os.listdir(path))

    save_model(path, functions)

    

def save_model(path, functions):
    from cloudinary.uploader import upload
    from cloudinary import config
    import libs.af_logins

    zip_filename = functions.get_model_folder_name()
    public_name = functions.get_model_folder_name()
    folder_name = functions.get_model_folder_name()

    print(os.listdir(path))

    shutil.make_archive(zip_filename, 'zip', f"{path}")
    response = upload(f'{folder_name}.zip', 
                      public_id=f'{public_name}', 
                      resource_type='auto')
    
    print("Response", response)
     

with DAG(
    default_args=default_args,
    dag_id="dag_for_wqm_v01",
    start_date=datetime(2023, 7, 1),
    schedule_interval='@daily' # 0 0 1,14 * * At 12:00 AM, on day 1 and 14 of the month
) as dag:
    task1 = PythonOperator(
        task_id='download_models',
        python_callable=download_models
    )
    
    task2 = PythonOperator(
        task_id='retrain_models',
        python_callable=retrain_models
    )

    task1 >> task2