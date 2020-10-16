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
    Accepts dataframe and cutoff value. Returns datframe with a new column containing upper outlier data for every numeric column.
    '''
    # iterate through numeric data type columns
    for col in df.select_dtypes('number'):

        # create column that contains values produced by upper_outliers function
        df[col + '_upper_outliers'] = upper_outliers(df[col], k)

    # return df
    return df

def upper_outlier_data_print(df):
    """
    Accepts dataframe. Returns .describe info for every column ending in "upper_outliers". To be used after add_upper_outlier_columns function.
    """
    # create list of column names that end with "_upper_outliers"
    upper_outlier_cols = [col for col in df if col.endswith('_upper_outliers')]
    
    # iterate through column name list
    for col in upper_outlier_cols:
        
        # print .describe info for each column from list
        print('~~~\n' + col)
        data = df[col][df[col] > 0]
        print(data.describe())

def outlier_remover(train):
    """
    Accepts a dataframe. Drops all rows with upper outliers identified by upper_outliers function.
    """
    # dropping columns where a value greater than 0 is found in the outlier column
    train.drop(train[train['bathroomcnt_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['bedroomcnt_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['calculatedfinishedsquarefeet_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['lotsizesquarefeet_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['taxvaluedollarcnt_upper_outliers'] > 0].index, inplace = True) 
    train.drop(train[train['taxamount_upper_outliers'] > 0].index, inplace = True) 
    # returning dataframe
    return train

def data_scaler(train, validate, test):
    """
    Accepts 3 dataframes: train, validate, test. Returns dataframes with non-categorical numerical columns scaled.
    """
    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to train column and scaling after
    train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.fit_transform(train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])

    # scaling data in dataframes
    validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    
    # return data frames
    return train, validate, test

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

def column_renamer(train, validate, test):
    """
    Accepts 3 dataframes. Returns them with various columns renamed.
    """
    # Renaming columns in given dataframe
    train.rename(columns = {'bathroomcnt': 'bathroom_count', 'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    validate.rename(columns = {'bathroomcnt': 'bathroom_count', 'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    test.rename(columns = {'bathroomcnt': 'bathroom_count', 'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)

    # Returning dataframes
    return train, validate, test

def final_prep():
    """
    No arguments needed. Function retrieves and prepares Zillow data for exploration with all changes from prep phase. 
    The returned dataset is an copy of the dataset you'd receive at the end of the prep phase.
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
    
    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to train column and transforming it 
    train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.fit_transform(train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])

    # scaling data in other dataframes
    validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])

    # listing features we're keeping from RFE ranking
    kept_features = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'heating_system_type_2', 'heating_system_type_7', 'heating_system_type_20', 'logerror']

    # assigning kept features our datasets
    train = train[kept_features]
    validate = validate[kept_features]
    test = test[kept_features]
    
    # Renaming columns in given dataframe
    train.rename(columns = {'bathroomcnt': 'bathroom_count', 'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    validate.rename(columns = {'bathroomcnt': 'bathroom_count', 'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)
    test.rename(columns = {'bathroomcnt': 'bathroom_count', 'bedroomcnt': 'bedroom_count', 'calculatedfinishedsquarefeet': 'property_sq_ft', 'taxvaluedollarcnt': 'tax_dollar_value', 'logerror': 'log_error'}, inplace=True)

    return train, validate, test