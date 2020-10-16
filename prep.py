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

def compare_column_values(df):
    """
    Accepts dataframe. Compares various bathroom and property square feet columns to determine how many unique values exist between them.
    Is used to check if columns are duplicates or near-duplicates.
    """
    # sum total number of unique values between pairs of columns
    sqft_columns_diff = (df.finishedsquarefeet12 != df.calculatedfinishedsquarefeet).sum()
    bathroom_count_diff = (df.calculatedbathnbr != df.bathroomcnt).sum()
    bathroom_count_diff_alt = (df.fullbathcnt != df.bathroomcnt).sum()

    # print results
    print(f'Number of different values between finishedsquarefeet12 and calculatedfinishedsquarefeet: {sqft_columns_diff}')
    print(f'Number of different values between calculatedbathnbr and bathroomcnt: {bathroom_count_diff}')
    print(f'Number of different values between fullbathcnt and bathroomcnt: {bathroom_count_diff_alt}')

def tax_columns_calculator(df):
    """
    Accepts dataframe. Prints avg % of rows where the sum of structuretaxvaluedollarcnt and landtaxvaluedollarcnt is equal to taxvaluedollarcnt.
    Is used to confirm if taxvaluedollarcnt column holds sums of other taxvalue columns.
    """
    # creating new df that holds all three columns we're interested in
    tax_eval_df = df[['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt']]

    # creating new column in df that is the sum of landtaxvaluedollarcnt and structuretaxvaluedollarcnt
    tax_eval_df['taxvaluedollarcnt_test'] = df.structuretaxvaluedollarcnt + df.landtaxvaluedollarcnt

    # comparing taxvaluedollarcnt to our manually calculated column and finding the average % of rows where the values matched
    print((tax_eval_df.taxvaluedollarcnt_test == tax_eval_df.taxvaluedollarcnt).mean())