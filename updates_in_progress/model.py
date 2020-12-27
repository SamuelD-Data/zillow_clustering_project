from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def baseline_function(train):
    """
    Function accepts 1 dataframe. Predicts logerror as the average logerror in every row then prints RMSE value of these predictions.
    """
    # creating empty dataframe
    y_train = pd.DataFrame()

    # adding log_error column to empty df
    y_train['log_error'] = train['log_error']

    # make baseline prediction equal to logerror mean
    y_train['baseline_pred'] = y_train['log_error'].mean()

    # evaluate rmse and print results
    rmse_train_bl = round(mean_squared_error(y_train.log_error, y_train.baseline_pred)**(1/2),6)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train_bl)

def model_1_function(train, predict):
    """
    Accepts 2 dataframes. 
    Train is the train df that model 1 will fit to. 
    Predict is the DF you would like model 1 to predict the log_error of after fitting to train.
    Prints RMSE of log_error predictions vs actual log_error on predict DF.
    """
    # making copies of train so we don't alter the original
    cluster_df = predict.copy()
    train_df = train.copy()
    
    # creating df's with the two features we want to cluster for each passed df
    X = cluster_df[['bedroom_count','property_sq_ft']]
    train_cluster_features = train_df[['bedroom_count','property_sq_ft']]

    # creating kmeans object and fitting to train data
    kmeans = KMeans(n_clusters = 3, random_state=123)
    kmeans.fit(train_cluster_features)

    # creating clusters and adding as columns
    cluster_df['cluster'] = kmeans.predict(X)
    train_df['cluster'] = kmeans.predict(train_cluster_features)
    
    # creating dummy dfs using cluster columns
    dummy_df = pd.get_dummies(cluster_df['cluster'])
    train_dummy_df = pd.get_dummies(train_df['cluster'])
    
    # renaming dummy columns 
    dummy_df.rename(columns = {0: 'cluster_0', 1: 'cluster_1', 2: 'cluster_2'}, inplace=True)
    train_dummy_df.rename(columns = {0: 'cluster_0', 1: 'cluster_1', 2: 'cluster_2'}, inplace=True)
    
    # adding dummy DFs to original DFs
    cluster_df = pd.concat([cluster_df, dummy_df], axis = 1)
    train_df = pd.concat([train_df, train_dummy_df], axis = 1)
    
    # dropping column dummy data is based on
    cluster_df.drop(columns=['cluster'] , inplace = True)
    train_df.drop(columns=['cluster'] , inplace = True)
    
    # select features for model 1 predictions
    Xfeat = cluster_df[['bedroom_count', 'property_sq_ft', 'tax_dollar_value', 'cluster_0', 'cluster_1', 'cluster_2']]
    yfeat = pd.DataFrame(cluster_df['log_error'])

    # creating linear regression object
    lm = LinearRegression(normalize=True)

    # fitting model to train data
    lm.fit(train_df[['bedroom_count', 'property_sq_ft', 'tax_dollar_value', 'cluster_0', 'cluster_1', 'cluster_2']], train_df['log_error'])

    # predict logerror on predict DF with model 1
    yfeat['model_1_pred'] = lm.predict(Xfeat)

    # evaluate RMSE and print results
    rmse_m1 = round(mean_squared_error(yfeat.log_error, yfeat.model_1_pred)**(1/2),6)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_m1)

def model_2_function(train, predict):
    """
    Accepts 2 dataframes. 
    Train is the train df that model 2 will fit to. 
    Predict is the df you would like model 2 to predict the log_error of after fitting to train.
    Prints RMSE of log_error predictions vs actual log_error.
    """
    # making copy of train so we don't alter the original
    predict_df = predict.copy()

    # select features for model 2 predictions
    Xfeat = predict_df[['bedroom_count', 'property_sq_ft', 'tax_dollar_value']]
    yfeat = pd.DataFrame(predict_df['log_error'])

    # creating linear regression object
    lm = LinearRegression(normalize=True)

    # fitting model to train data
    lm.fit(train[['bedroom_count', 'property_sq_ft', 'tax_dollar_value']], train['log_error'])

    # predict logerror with model 2
    yfeat['model_2_pred'] = lm.predict(Xfeat)

    # evaluate RMSE and print results
    rmse_m2 = round(mean_squared_error(yfeat.log_error, yfeat.model_2_pred)**(1/2),6)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_m2)

def model_3_function(train, predict):
    """
    Accepts 2 dataframes. 
    Train is the train df that model 3 will fit to. 
    Predict is the DF you would like model 3 to predict the log_error of after fitting to train.
    Prints RMSE of log_error predictions vs actual log_error on predict DF.
    """
    # making copies of train so we don't alter the original
    cluster_df = predict.copy()
    train_df = train.copy()
    
    # creating df's with the two features we want to cluster for each passed df
    X = cluster_df[['bedroom_count','property_sq_ft']]
    train_cluster_features = train_df[['bedroom_count','property_sq_ft']]

    # creating kmeans object and fitting to predict data
    kmeans = KMeans(n_clusters = 3, random_state=123)
    kmeans.fit(X)

    # creating clusters and adding as columns
    cluster_df['cluster'] = kmeans.predict(X)
    train_df['cluster'] = kmeans.predict(train_cluster_features)
    
    # creating dummy dfs using cluster columns
    dummy_df = pd.get_dummies(cluster_df['cluster'])
    train_dummy_df = pd.get_dummies(train_df['cluster'])
    
    # renaming dummy columns 
    dummy_df.rename(columns = {0: 'cluster_0', 1: 'cluster_1', 2: 'cluster_2'}, inplace=True)
    train_dummy_df.rename(columns = {0: 'cluster_0', 1: 'cluster_1', 2: 'cluster_2'}, inplace=True)
    
    # adding dummy DFs to original DFs
    cluster_df = pd.concat([cluster_df, dummy_df], axis = 1)
    train_df = pd.concat([train_df, train_dummy_df], axis = 1)
    
    # dropping column dummy data is based on
    cluster_df.drop(columns=['cluster'] , inplace = True)
    train_df.drop(columns=['cluster'] , inplace = True)
    
    # select features for model 3 predictions
    Xfeat = cluster_df[['cluster_0', 'cluster_1', 'cluster_2']]
    yfeat = pd.DataFrame(cluster_df['log_error'])

    # creating linear regression object
    lm = LinearRegression(normalize=True)

    # fitting model to train data
    lm.fit(train_df[['cluster_0', 'cluster_1', 'cluster_2']], train_df['log_error'])

    # predict logerror on predict DF with model 3
    yfeat['model_3_pred'] = lm.predict(Xfeat)

    # evaluate RMSE and print results
    rmse_m3 = round(mean_squared_error(yfeat.log_error, yfeat.model_3_pred)**(1/2),6)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_m3)