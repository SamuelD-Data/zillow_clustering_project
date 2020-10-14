from env import host, user, password

import pandas as pd
import numpy as np
import os

def get_connection(db, user=user, host=host, password=password):
    """
    Function returns a URL string that can be used to connect to the data science database.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    """
    Function connects to the data science database and returns a data frame containing Zillow data for 2017 properties and predictions. 
    """
    # create SQL query string
    sql_query = "SELECT * FROM properties_2017 JOIN predictions_2017 on predictions_2017.parcelid = properties_2017.parcelid WHERE unitcnt = 1"
    
    # creates dataframe using data from DS database, Zillow table
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    # writes data to csv file for future use
    df.to_csv('zillow_df.csv')
    
    # returns data frame
    return df