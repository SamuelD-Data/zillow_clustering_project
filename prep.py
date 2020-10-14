# import modules needed to run functions
import pandas as pd
import numpy as np
import sklearn
from acquire import get_zillow_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def missing_rows(df):
    """
    Function accepts data frame and returns a dataframe that shows the total number and percent of each column's values that are null.
    """
    # taking sum of missing rows for each variable, multiplying by 100 then dividing by total 
    # number of rows in original df to find % of missing rows 
    missing_row_percent = df.isnull().sum() * 100 / len(df)
    # count number of missing values for each variable and sum for each
    missing_row_raw = df.isnull().sum()
    # creating df using series' created by 2 previous variables
    missing_df = pd.DataFrame({'num_rows_missing' : missing_row_raw, 'pct_rows_missing': missing_row_percent})
    # return df
    return missing_df


def drop_missing_columns(df):
    """
    Accepts a dataframe and removes any columns that are missing 40% or more of their values, then returns dataframe.
    """
    # setting percent of values in each column that must be non-null in order for column to not be dropped
    prop_required_column = .6
    threshold = int(round(prop_required_column*len(df.index),0))
    # dropping columns
    df.dropna(axis=1, thresh=threshold, inplace=True)
    # returning df
    return df

def drop_selected_columns(df):
    """
    Accepts dataframe and drops all categorical columns with more than 100 unique values or only 1 unique value.
    """
    df.drop(columns=['id', 'parcelid', 'latitude', 'longitude', 'propertycountylandusecode', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock', 'transactiondate', 'assessmentyear', 'unitcnt', 'finishedsquarefeet12', 'calculatedbathnbr', 'fullbathcnt', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt'] , inplace = True)




