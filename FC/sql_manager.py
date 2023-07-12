import pandas as pd
import functions


from sqlalchemy import create_engine
from sqlalchemy import text
import logins

import file_management

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def get_creds_local():
    server = 'localhost' 
    database = 'wqm' 
    username = 'wqm_admin' 
    password = 'password'  
    port = 3306

    con = f'mysql+pymysql://{username}:{password}@{server}/{database}'
    return con

def get_creds_cloud():
    server = logins.server
    database =  logins.database
    username =  logins.username
    password = logins.password  
    port = 3306

    con = f'mysql+pymysql://{username}:{password}@{server}/{database}'
    return con

def get_creds_lake():
    server = logins.server
    database =  logins.database_lake
    username =  logins.username
    password = logins.password  
    port = 3306

    con = f'mysql+pymysql://{username}:{password}@{server}/{database}'
    return con 

def connect():
    engine = create_engine(
            get_creds_local(), 
            pool_recycle=3600)
    print(get_creds_cloud())
    return engine

def connect_lake():
    engine = create_engine(
            get_creds_lake(), 
            pool_recycle=3600)
    print(get_creds_lake())
    return engine

def get_table_name():
    return functions.get_table_name()

def check_tables(engine, table):
    isTable = False

    query = text(f"SELECT * FROM {table}")

    with engine.begin() as conn:
        try:
            result = conn.execute(query)
        except:
            return isTable 
        
    isTable = True
    return result

def remove_table(table, engine):
    query = text(f"Drop table {table}")
    with engine.begin() as conn:
        try:
            result = conn.execute(query)
        except:
            return False


def pandas_to_sql(table_name, pandas_dataset, engine):
    pandas_dataset.to_sql(table_name, con=engine)
    
def pandas_to_sql_if_exists(table_name, pandas_dataset, engine, action):
    pandas_dataset.to_sql(table_name, con=engine, if_exists=action)


def sql_to_pandas(table_name, engine):
    output = pd.read_sql_table(table_name, con=engine.connect())
    try:
        output = output.drop(["index"], axis = 1)
        print(f"'index' parameter dropped {table_name}");
    except:
        print("'index' parameter does not exist");
    
    try:
        output = output.drop(["level_0"], axis = 1)
        print("'level_0' parameter dropped");
    except:
        print("'level_0' parameter does not exist");
    
    return output



def get_vals(table, col, val, engine):
    # query = text(f"SELECT time, current, integrals, Concentration FROM {table} where `{col}` = {val}")
    query = text(f"SELECT * FROM {table} where `{col}` = {val}")
    return get_query_to_pandas(engine, query)
    
def get_two_vals(engine, table, col1, val1, col2, val2):
    # query = text(f"SELECT time, current, integrals, Concentration FROM {table} where `{col1}` = {val1} and `{col2}` = {val2}")
    query = text(f"SELECT * FROM {table} where `{col1}` = {val1} and `{col2}` = {val2}")
    return get_query_to_pandas(engine, query)

def get_query_to_pandas(engine, query):
    with engine.begin() as conn:
            result = conn.execute(query)

    output = pd.DataFrame()
    for r in result:

        df_dictionary = pd.DataFrame([r._asdict()])
        output = pd.concat([output, df_dictionary], ignore_index=True)

    try:
        output = output.drop(["index"], axis = 1)
    except:
        pass
    
    try:
        output = output.drop(["level_0"], axis = 1)
    except:
        pass
    
    return output
