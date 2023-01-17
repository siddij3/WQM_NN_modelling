# In[12]:


import csv
import pandas as pd
import os


def transform_data(csvfile):

  df = pd.read_csv(csvfile, header=None)

  sensorNum = df.values.tolist()[0][1:]
  pH_raw = df.values.tolist()[1][1:]
  starting = df.values.tolist()[2][1:]
  increasing = df.values.tolist()[3][1:]
  decreasing = df.values.tolist()[4][1:]
  temperature = df.values.tolist()[5][1:]
  days_elapsed = df.values.tolist()[6][1:]
  sensor_cycle = df.values.tolist()[7][1:]
  repeat_use = df.values.tolist()[8][1:]
  measurements_1 = df.values.tolist()[9][1:]
  measurements_cont = df.values.tolist()[10][1:]

  pH_values = [x for x in pH_raw if str(x) != 'nan' and 'omit' not in str(x).lower()]

  pH = []

  for i in pH_values:
    try:
      pH.append(float(i))
    except:
      pass

  exp_params = df.values.tolist()
  params = []

  for i in range (0, 11):
      params.append(exp_params[i][0])

  df2 = df[11:].dropna(axis='columns')
  
  k=0
  times=[]
  voltage=[]
  amps=[]

  for i,j in df2.iterrows():
    curr = list(map(float,j.values))

    times.append(curr[0])

    voltage.append(curr[1:])

    for i in curr[1:]:
      amps.append(i)

  print(len(amps))
  length=len(voltage[18])


  durations = []
  n=-1
  for i in range(len(amps)):
    amps[i]
    if i%length==0:
      n+=1

    durations.append(times[n])
  start = []
  increase = []
  decrease = []

  pH = []
  sensor_number = []
  Temp = []
  repeat = []
  cycle = []
  days = []
  m_1 = []
  m_cont = []

  m=0

  for i in range(len(amps)):
    if(m==len(pH_values)): 
      m=0

    pH.append(pH_values[m])

    sensor_number.append(sensorNum[m])
    start.append(starting[m])
    increase.append(increasing[m])
    decrease.append(decreasing[m])

    Temp.append(temperature[m])

    days.append(days_elapsed[m])
    cycle.append(sensor_cycle[m])
    repeat.append(repeat_use[m])
    m_1.append(measurements_1[m])
    m_cont.append(measurements_cont[m])
    
    m+=1
  
  dict3 = { 'Time':durations,
            'Voltage':amps, 
            'Days Elapsed':days ,
            'Sensor Cycle':cycle, 
            'Start':start, 
            'Increasing':increase, 
            'Decreasing':decrease, 
            'Temperature': Temp,
            'Repeat Use':repeat , 
            'Sensor Number': sensor_number,
            'Measurement Number':m_1,
            'Measurement Continuous':m_cont,
            'pH': pH
            }
  df_final = pd.DataFrame(dict3)
  
  

  columns = df_final.columns.to_list()
  print(columns)


  return df_final


filepath = r".\\"
local_download_path = os.path.expanduser(filepath)
filenames=[]
for filename in os.listdir(local_download_path):
  if filename.endswith('csv') and 'Entries' in filename:
    filenames.append(filename)

sum = 0
for i in range(len(filenames)):
  df = transform_data(filenames[i])
  sum+=len(df)

  
if len(filenames)>1:
  for i in filenames[1:]:
    #print('File appended: '+i)
    df= df.append(transform_data(i),ignore_index=True,sort=False)
df.to_csv('.\\Data\\aggregated_data_pH.csv',index=False)


import numpy as np
#for i in df.to_numpy():
#    if (np.isnan(i).any()):
      #  print(i)

