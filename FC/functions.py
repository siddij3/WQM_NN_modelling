from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import logins

import cloudinary
import cloudinary.api
import requests
import shutil
import os
import file_management

from pandas import DataFrame

def get_dict(tt_feats):
    dict = {
    'Time':tt_feats[:, 0],
    'Current':tt_feats[:, 1], 
    'pH Elapsed':tt_feats[:, 2] ,
    'Temperature':tt_feats[:, 3], 
    'Rinse':tt_feats[:, 4],
    'Integrals':tt_feats[:, 5]
    }
    return DataFrame(dict)

def get_model_folder_name():
    return "gold_fc_cloud"

def get_table_name():
    return 'gold_fc'

def get_std_params_table_name():
    return 'std_params_gold'

def get_data_folder_name():
    return "Data"

def load_from_cloud():

    # Specify the public ID or the URL of the ZIP file to download
    zip_public_id = f'{get_model_folder_name()}.zip'  # Or provide the URL of the ZIP file
    # Get the download URL for the ZIP file from Cloudinary
    download_url = cloudinary.utils.cloudinary_url(zip_public_id, 
                                                resource_type='raw')[0]

    response = requests.get(download_url, stream=True)

    save_path = f".\\{zip_public_id}"

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    #Unzip file
    path = os.path.expanduser(file_management.get_file_path())
    shutil.unpack_archive(save_path, path)

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

def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
