from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import sklearn


def split_data(df, target):
    """
    Accepts DF and target variable name. Returns data split into 6 dataframes: y_train, y_validate, y_test, X_train, X_validate, X_test.
    """
    # splitting data
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # specifying which columns to keep in outputted dataframe
    # x = features | y = target variable
    #X_train = train.drop(columns=[target])
    #y_train = train[[target]]
    
    #X_validate = validate.drop(columns=[target])
    #y_validate = validate[[target]]
    
    #X_test = test.drop(columns=[target])
    #y_test = test[[target]]
    
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
    # returning series described in doc string
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Accepts dataframe and cutoff value. Returns datframe with a new column containing upper outlier data for every numeric column.
    '''
    # iterate through numeric data type columns
    for col in df.select_dtypes('number'):
        # create column with column name and "upper_outliers" added to it
        # column contains values produced by upper_outliers function
        df[col + '_upper_outliers'] = upper_outliers(df[col], k)
    return df

def upper_outlier_data_print(df):
    """
    Accepts dataframe. Returns .describe info for every column ending in "upper_outliers". To be used after add_upper_outlier_columns function.
    """
    upper_outlier_cols = [col for col in df if col.endswith('_upper_outliers')]
    for col in upper_outlier_cols:
        print('~~~\n' + col)
        data = df[col][df[col] > 0]
        print(data.describe())

def data_scaler(train, validate, test):
    """
    Accepts 3 dataframes: train, validate, test. Returns dataframes with non-categorical numerical columns scaled.
    """
    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to train column
    train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.fit_transform(train[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])

    # scaling data in dataframes
    validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(validate[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']] = scaler.transform(test[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt', 'taxamount']])
    
    # return data frames
    return train, validate, test
