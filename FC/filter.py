import functions
import sql_manager
import numpy as np

if __name__ == '__main__':
    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect_lake()
    dataset= sql_manager.get_vals(table_name, "Rinse", 1, engine)
    
    a = np.array_split(dataset, len(dataset)/51)
    for i in a:
        filtered_df = functions.filterData(i)
   
        print(filtered_df)

    