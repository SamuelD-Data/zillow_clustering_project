from env import host, user, password

import pandas as pd
import numpy as np
import os

def get_connection(db, user=user, host=host, password=password):
    """
    Function returns a URL string that can be used to connect to the data science database.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# creating function that creates a variable call filename that holds the name of a current, or soon to be created file
def get_zillow_data():
    
    # create SQL query string
    sql_query = "SELECT * FROM properties_2017 JOIN predictions_2017 on predictions_2017.parcelid = properties_2017.parcelid WHERE unitcnt = 1"
    
    # set filename for reference
    filename = "zillow.csv"
    
    # if a file is found with a name that matches filename (zillow.csv) 
    # return the data as a dataframe
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if no file with the specified name can be found
    else:
    
        # create dataframe using data from DS database, Zillow table
        df = pd.read_sql(sql_query, get_connection('zillow'))

        # writing dataframe to csv file
        df.to_csv(filename, index = False)

        # return the dataframe
        return df 