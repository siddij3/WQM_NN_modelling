import os
import pandas as pd
import numpy as np
import sql_manager

def get_data_path():
    folder_name = "Data"
    return f".\\{folder_name}"

if __name__ == '__main__':
    filepath = get_data_path()
    local_download_path = os.path.expanduser(filepath)
    filenames=[]
    i = 0
    for filename in os.listdir(local_download_path):
        if filename.endswith('csv') and "raw" in filename:
            filenames.append(filepath + "\\" + filename)

    for i in range(len(filenames)):
        df = pd.read_csv(filenames[i], header=None)

    num_tests = int((df.shape[0]-1)/51)

    all_tests = []

    for i in range(num_tests):
        test = [None]*51    


        for j in range(51):

            loc_index = 1

            while True:
                if int(df.iloc[loc_index][0]) == j:
                    test[j] = df.iloc[loc_index]
                    df = df.drop(df.index[loc_index])
                    break

                loc_index += 1
        all_tests.append(test)
        
    x_vals = []

    y_integrals = []
    pH = []
    rinse = []
    temperature = []
    current = []
    concentration = []
    all_integrals = []

    diffs = []
    longest_range_for_diff = 26

    for test in all_tests:
        y_vals = []

        # Chosen based on measurements
        for second in test:
            x_vals.append(float(second[0])) 
            current.append(float(second[1]))
            pH.append(float(second[2]))
            temperature.append(float(second[3]))
            rinse.append(int(second[4]))
            concentration.append(float(second[5]))

    for i,x_val in enumerate(x_vals):
        if int(x_val) == 0:
            y_integrals.append(np.trapz(current[i:i+longest_range_for_diff], x_vals[i:i+longest_range_for_diff]))

    print(y_integrals)

    m = 0
    for i in range(len(y_integrals)):
        for j in range(51):

            all_integrals.append(y_integrals[i])

    
    print(all_integrals)


    dict = {
    'Time':x_vals,
    'Current':current, 
    'pH Elapsed': pH,
    'Temperature': temperature, 
    'Rinse':  rinse,
    'Integrals': all_integrals,
    'Concentration': concentration

    }   


    df_final = pd.DataFrame(dict)

    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect()

    sql_manager.pandas_to_sql(table_name, df_final, engine)


