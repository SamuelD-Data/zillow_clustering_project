# import modules needed to run functions
# imports
from acquire import get_zillow_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

import pandas as pd
import numpy as np
import sklearn

import warnings
warnings.filterwarnings("ignore")

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
    df.drop(columns=['id', 'parcelid', 'latitude', 'longitude', 'propertycountylandusecode', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock', 'transactiondate', 'assessmentyear', 'unitcnt', 'finishedsquarefeet12', 'calculatedbathnbr', 'fullbathcnt', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'buildingqualitytypeid', 'propertylandusetypeid', 'id.1', 'parcelid.1'] , inplace = True)
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

def split_data(df):
    """
    Accepts DF. Returns data split into 3 dataframes: train, validate, and test.
    """
    # splitting data
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # return split data frames
    return train, validate, test

def upper_outliers(s, k):
    '''
    Accepts series and cutoff value.
    If a value in the series is an upper outlier, it returns a number that represents how far above the value is from the upper bound
    or 0 if the number is not an outlier.
    '''
    # creating 2 variables that represent the 1st and 3rd quantile of the given series
    q1, q3 = s.quantile([.25, .75])

    # calculating IQR
    iqr = q3 - q1

    # calculating upper bound
    upper_bound = q3 + k * iqr

    # returning series 
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Accepts dataframe and cutoff value. 
    Returns datframe with a new column containing upper outlier data for every numeric column.
    '''
    # iterate through numeric data type columns
    for col in df.select_dtypes('number'):

        # create column that contains values produced by upper_outliers function
        df[col + '_upper_outliers'] = upper_outliers(df[col], k)

    # return df
    return df

# creating functions to identify upper outliers and show how far below the lower bound they are

def lower_outliers(s, k):
    '''
    Accepts series and cutoff value.
    If a value in the series is an lower outlier, it returns a number that represents how far above the value is from the lower bound
    or 0 if the number is not an outlier.
    '''
    # creating 2 variables that represent the 1st and 3rd quantile of the given series
    q1, q3 = s.quantile([.25, .75])

    # calculating IQR
    iqr = q3 - q1

    # calculating lower bound
    lower_bound = q1 - k * iqr

    # returning series 
    return s.apply(lambda x: max([lower_bound - x, 0]))

def add_lower_outlier_columns(df, k):
    '''
    Accepts dataframe and cutoff value. Returns datframe with a new column containing lower outlier data for every numeric column.
    '''
    # iterate through numeric data type columns
    for col in df.select_dtypes('number'):
        if col.endswith('_upper_outliers') == False:

            # create column that contains values produced by lower_outliers function
            df[col + '_lower_outliers'] = lower_outliers(df[col], k)

    # return df
    return df

def outlier_remover(df):
    """
    Accepts dataframe. Drops any row with a value > 0 in a column ending in "_outliers". 
    """
    # create list of column names that end with "_outliers"
    outlier_cols = [col for col in df if col.endswith('_outliers')]
    
    # iterate through column name list
    for col in outlier_cols:
        df.drop(df[df[col] > 0].index, inplace = True) 
    return df

def handle_outliers(df, k):
    """
    Accepts DF and cutoff value. Identifies and removes all outliers with respect to given cutoff value. 
    Removes all "outliers" columns created by function.
    """
    # passing given df and cutoff value to functions that will add "upper outlier" and "lower outlier" columns
    add_upper_outlier_columns(df, k)
    add_lower_outlier_columns(df, k)
    
    # passing df to function that removes outliers identified by previous functions
    outlier_remover(df)

    # returning df
    return df

def data_scaler(train, validate, test):
    """
    Accepts 3 dataframes: train, validate, test. Returns dataframes with non-categorical numerical columns scaled.
    """
    # copying passed dataframes so we don't alter the originals
    # we will need both scaled and unscaled data in our project
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to train column and scaling after
    train_scaled[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.fit_transform(train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])

    # scaling data in dataframes
    validate_scaled[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    test_scaled[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    
    # return data frames
    return train_scaled, validate_scaled, test_scaled

def rfe_ranker(train):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a linear regression model.
    """
    # creating linear regression object
    lm = LinearRegression()

    # fitting linear regression model to features 
    lm.fit(train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount', 'heating_system_type_2', 'heating_system_type_7', 'heating_system_type_20']], train['logerror'])

    # creating recursive feature elimination object and specifying to only rank 1 feature as best
    rfe = RFE(lm, 1)

    # using rfe object to transform features 
    x_rfe = rfe.fit_transform(train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount', 'heating_system_type_2', 'heating_system_type_7', 'heating_system_type_20']], train['logerror'])

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating train df for rfe object 
    rfe_train = train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount', 'heating_system_type_2', 'heating_system_type_7', 'heating_system_type_20']]

    # creating list of the top features per rfe
    rfe_features = rfe_train.loc[:,feature_mask].columns.tolist()

    # creating ranked list 
    feature_ranks = rfe.ranking_

    # creating list of feature names
    feature_names = rfe_train.columns.tolist()

    # create df that contains all features and their ranks
    rfe_ranks_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})

    # return df sorted by rank
    return rfe_ranks_df.sort_values('Rank')

def rfe_column_dropper(train, validate, test, train_scaled, validate_scaled, test_scaled):
    """
    Accepts 3 unscaled and 3 scaled dataframe, 6 totals. Returns them only the 3 features selected based on RFE and logerror.
    """
    # listing features we're keeping from RFE ranking
    kept_features = ['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'logerror']

    # assigning kept features our datasets
    train_scaled = train_scaled[kept_features]
    validate_scaled = validate_scaled[kept_features]
    test_scaled = test_scaled[kept_features]

    train = train[kept_features]
    validate = validate[kept_features]
    test = test[kept_features]
    
    # returning data frames
    return train, validate, test, train_scaled, validate_scaled, test_scaled

def column_renamer(unscaled_train, unscaled_validate, unscaled_test, scaled_train, scaled_validate, scaled_test):
    """
    Accepts 3 unscaled and 3 scaled dataframe, 6 totals. Returns them will all columns renamed.
    """
    # Renaming columns in given dataframe
    unscaled_train.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    unscaled_validate.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    unscaled_test.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)

    scaled_train.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    scaled_validate.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    scaled_test.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)

    # Returning dataframes
    return unscaled_train, unscaled_validate, unscaled_test, scaled_train, scaled_validate, scaled_test

def final_prep():
    """
    No arguments needed. Function returns train, validate and test Zillow datasets ready for exploration with all changes from prep phase. 
    """
    # create variable that will hold DF for easy access to data
    df = get_zillow_data()
    
    # filter dataset to only keep best RFE ranked columns
    df = df[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet','lotsizesquarefeet', 'taxvaluedollarcnt','taxamount', 'heatingorsystemtypeid', 'logerror']]
    
    # drop all rows with missing values
    df.dropna(inplace = True)
    
    # creating dummy df using heatingorsystemtypeid column
    dummy_df = pd.get_dummies(df['heatingorsystemtypeid'])
    
    # renaming dummy columns 
    dummy_df.rename(columns = {2.0: 'heating_system_type_2', 7.0: 'heating_system_type_7', 20.0: 'heating_system_type_20'}, inplace=True)
    
    # adding dummy df to original df
    df = pd.concat([df, dummy_df], axis = 1)
    
    # dropping column dummy data is based on
    df.drop(columns=['heatingorsystemtypeid'] , inplace = True)
    
    # splitting data
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # iterate through numeric data type columns
    for col in train.select_dtypes('number'):

        # create column that contains values produced by upper_outliers function
        train[col + '_upper_outliers'] = upper_outliers(train[col], 6)
    
    # dropping columns where a value greater than 0 is found in the outlier column
    train.drop(train[train['bathroomcnt_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['bedroomcnt_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['calculatedfinishedsquarefeet_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['lotsizesquarefeet_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['taxvaluedollarcnt_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['taxamount_upper_outliers'] > 0].index, inplace = True) 
    
    # creating list of outlier column names
    outlier_cols = [col for col in train if col.endswith('_outliers')]

    # dropping each column from list of outlier columns
    train = train.drop(columns = outlier_cols)

    # creating copies of each df so we can scale one set and leave another set unscale
    unscaled_train = train.copy()
    unscaled_validate = validate.copy()
    unscaled_test = test.copy()

    scaled_train = train.copy()
    scaled_validate = validate.copy()
    scaled_test = test.copy()

    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to train column and transforming it 
    scaled_train[['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']] = scaler.fit_transform(scaled_train[['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']])

    # scaling data in other dataframes
    scaled_validate[['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']] = scaler.transform(scaled_validate[['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']])
    scaled_test[['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']]= scaler.transform(scaled_test[['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']])

    # listing features we're keeping from RFE ranking
    kept_features = ['bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'logerror']

    # assigning kept features our datasets
    unscaled_train = unscaled_train[kept_features]
    unscaled_validate = unscaled_validate[kept_features]
    unscaled_test = unscaled_test[kept_features]

    scaled_train = scaled_train[kept_features]
    scaled_validate = scaled_validate[kept_features]
    scaled_test = scaled_test[kept_features]
    
    # Renaming columns in given dataframe
    unscaled_train.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    unscaled_validate.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    unscaled_test.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)

    scaled_train.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    scaled_validate.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    scaled_test.rename(columns = {'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    
    # returning scaled and unscaled datasets
    return unscaled_train, unscaled_validate, unscaled_test, scaled_train, scaled_validate, scaled_test