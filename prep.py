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
    Accepts dataframe and drops all categorical columns with more than 10 unique values or only 1 unique value.
    """
    # dropping columns specified by column name
    df.drop(columns=['id', 'parcelid', 'latitude', 'longitude', 'propertycountylandusecode', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock', 'transactiondate', 'assessmentyear', 'unitcnt', 'finishedsquarefeet12', 'calculatedbathnbr', 'fullbathcnt', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'buildingqualitytypeid', 'propertylandusetypeid'] , inplace = True)
    # returning df
    return df

def drop_more_selected_columns(df):
    """
    Accepts dataframe and drops all categorical columns that only contain 1 unique value after all null values were removed.
    """
    # dropping columns specified by column name
    df.drop(columns=['fips', 'regionidcounty', 'roomcnt'] , inplace = True)
    return df

def zillow_dummy(df):
    """
    Accepts a data frame, returns it with heatingorsystemtypeid column split into 3 dummary variables columns and original heatingorsystemtypeid column removed.
    """
    # creating dummy df using heatingorsystemtypeid column
    dummy_df = pd.get_dummies(df['heatingorsystemtypeid'])
    # renaming dummy columns 
    dummy_df.rename(columns = {2.0: 'heating_system_type_2', 7.0: 'heating_system_type_7', 20.0: 'heating_system_type_20'}, inplace=True)
    # adding dummy df to original df
    df = pd.concat([df, dummy_df], axis = 1)
    # dropping column dummy data is based on
    df.drop(columns=['heatingorsystemtypeid'] , inplace = True)
    # returning df
    return df

