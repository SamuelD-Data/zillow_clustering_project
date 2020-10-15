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
    X_train = train.drop(columns=[target])
    y_train = train[[target]]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[[target]]
    
    X_test = test.drop(columns=[target])
    y_test = test[[target]]
    
    return y_train, y_validate, y_test, X_train, X_validate, X_test