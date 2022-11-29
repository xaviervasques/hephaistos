#!/usr/bin/python3
# time_series.py
# Author: Xavier Vasques (Last update: 05/04/2022)

# Copyright 2022, Xavier Vasques. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""

Features engineering requires the use of different methods to rescale continuous data and encode categorical data. If we add the time component, feature engineering can be more complex
to understand even if time-related features are found in many fields such as finance, weather forecast or healthcare. Time series is a sequence of numerical values representing the
evolution of a specific quantity over time. The data is captured at equal intervals. Time series is a good way to understand the behavior of a variable in the past and use this
knowledge to predict the future behavior of the variable through probability and statistic concepts. With the variable time, we can predict a stock price based on what happened
yesterday, predict the evolution of a disease based on past experiences or predict the road traffic in a city if we have data about the last few years. Time series may also bring
seasonality or trends that we can mathematically model.

"""

# Import necessary python libraries
import pandas as pd
import numpy as np

def split(time_split, time_feature_name, time_format, X, X_train, X_test):

        """
        We load the dataset (X, X_train, X_test) with Pandas and create a DataFrame with new columns (year, month, day, hour, minute, second) for each observation in the series. Of course, we can play with time variables to do more than just using year, month, or day alone. We can couple time information with other features to improve the performance of our models such as the season of the month or semester. For instance, if our time stamp has hours, and we want to study the road traffic, we can add some variables such as business hours and non-business hours or the name of the day in a week. DatetimeIndex from Pandas provides many attributes.

        The inputs of the split function are:

        time_split = ['year','month','day', 'hour', 'second'] / Select which variables you want (year, month, hours, minutes, seconds) https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html

        time_feature_name = 'date' (Name of the time feature)

        The strftime to parse time, e.g. "%d/%m/%Y". See strftime documentation for more information on choices: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        For exemple is the data is as follows '1981-7-1 15:44:31' a format would be "%Y-%d-%m %H:%M:%S"
        time_format = "%Y-%m-%d

        X, X_train, X_test: the dataset processed in main.py according to inputs.py guidance

        """
        # Split time information in X, X_train and X_test
        X[time_feature_name] = pd.to_datetime(X[time_feature_name],format=time_format)
        X_train[time_feature_name] = pd.to_datetime(X_train[time_feature_name],format=time_format)
        X_test[time_feature_name] = pd.to_datetime(X_test[time_feature_name],format=time_format)
        
        # Time information (year, month, day, minute, second).
        for split in time_split:
            if split == 'year':
                X[split]=X[time_feature_name].dt.year
                X_train[split]=X_train[time_feature_name].dt.year
                X_test[split]=X_test[time_feature_name].dt.year
            if split == 'month':
                X[split]=X[time_feature_name].dt.month
                X_train[split]=X_train[time_feature_name].dt.month
                X_test[split]=X_test[time_feature_name].dt.month
            if split == 'day':
                X[split]=X[time_feature_name].dt.day
                X_train[split]=X_train[time_feature_name].dt.day
                X_test[split]=X_test[time_feature_name].dt.day
            if split == 'hour':
                X[split]=X[time_feature_name].dt.hour
                X_train[split]=X_train[time_feature_name].dt.hour
                X_test[split]=X_test[time_feature_name].dt.hour
            if split == 'minute':
                X[split]=X[time_feature_name].dt.minute
                X_train[split]=X_train[time_feature_name].dt.minute
                X_test[split]=X_test[time_feature_name].dt.minute
            if split == 'second':
                X[split]=X[time_feature_name].dt.second
                X_train[split]=X_train[time_feature_name].dt.second
                X_test[split]=X_test[time_feature_name].dt.second
        
        return X, X_train, X_test

# Time series transformations / We can combine different methods
def lag(X, y, X_train, y_train, X_test, y_test, lagged_features=None, number_of_lags=None, lag_aggregation=None):
 
    """

    Let’s say we are predicting the stock price for a company. To make a prediction, we will consider the past values. The prediction of a value at time t will be impacted by the value at
    time t-1. We need to create features to represent these past values. We call lags the past values such as t-1 is lag one, t-2 is lag two, etc. We can use the shift() method in pandas to
    create the lags.

        In inputs.py, if we want to add lags to our data lag = 'yes' or 'no', how many (number_of_lags), which features (lagged_features), aggregation (lag_aggregation)
        For aggregation, the following options are avaible:
            - "no" if you do not need aggregation
            - "min", "max", "mean", "std"
        number_of_lags is the number of lags you want (set to 0 if 'no' was selected)
        
        X, y, X_train, y_train, X_test, y_test are the splitted dataset
        
    """

    for col_name in lagged_features:
        # Variable to store selected lag features to use for aggregation calculations
        lagged_feature_cols = []
        for x in range(1, number_of_lags + 1):
            lag_input = 'lag_%i_%s'%(x,col_name)
            X[lag_input] = X[col_name].shift(x)
            X_train[lag_input] = X_train[col_name].shift(x)
            X_test[lag_input] = X_test[col_name].shift(x)
            lagged_feature_cols = [lag_input] + lagged_feature_cols
        
        for var in lag_aggregation:
            # Aggregating features through statistics such as average, standard deviation, maximum, minimum of skewness might be valuable additions to predict future behavior. Pandas provides the aggregate method to do that.
            X_lagged_features = X.loc[:, lagged_feature_cols]
            X_train_lagged_features = X_train.loc[:, lagged_feature_cols]
            X_test_lagged_features = X_test.loc[:, lagged_feature_cols]
            # Create aggregated features
            if var == 'max':
                X['max_lag_%s'%col_name] = X_lagged_features.aggregate(np.max, axis=1)
                X_train['max_lag_%s'%col_name] = X_train_lagged_features.aggregate(np.max, axis=1)
                X_test['max_lag_%s'%col_name] = X_test_lagged_features.aggregate(np.max, axis=1)
            if var == 'min':
                X['min_lag_%s'%col_name] = X_lagged_features.aggregate(np.min, axis=1)
                X_train['min_lag_%s'%col_name] = X_train_lagged_features.aggregate(np.min, axis=1)
                X_test['min_lag_%s'%col_name] = X_test_lagged_features.aggregate(np.min, axis=1)
            if var == 'mean':
                X['mean_lag_%s'%col_name] = X_lagged_features.aggregate(np.mean, axis=1)
                X_train['mean_lag_%s'%col_name] = X_train_lagged_features.aggregate(np.mean, axis=1)
                X_test['mean_lag_%s'%col_name] = X_test_lagged_features.aggregate(np.mean, axis=1)
            if var == 'std':
                X['std_lag_%s'%col_name] = X_lagged_features.aggregate(np.std, axis=1)
                X_train['std_lag_%s'%col_name] = X_train_lagged_features.aggregate(np.std, axis=1)
                X_test['std_lag_%s'%col_name] = X_test_lagged_features.aggregate(np.std, axis=1)
    
    return X, X_train, X_test, y, y_train, y_test

# Create a rolling window mean feature
def rolling_window(X, X_train, X_test, rolling_features=None, window_size=None):

    """

    The rolling window features will calculate statistics by selecting a window size, take the average of the values in the selected window, and use the result as a feature. That’s why it
    is called rolling window (the generated features by the method) as the selected window is sliding with every next point.

    Create a rolling window mean feature (as defined in inputs.py):
        window_size is an integer indicating the size of the window,
        rolling_features to select the features we want to apply rolling window,
        X, X_train, X_test the dataset.
        
    """

    for col_name in rolling_features:
        X['rolling_window_%s'%col_name] = X[col_name].rolling(window=window_size).mean()
        X_train['rolling_window_%s'%col_name] = X_train[col_name].rolling(window=window_size).mean()
        X_test['rolling_window_%s'%col_name] = X_test[col_name].rolling(window=window_size).mean()
    return X, X_train, X_test

# Create an expending rolliing window mean feature
def expending_window(X, X_train, X_test, expending_features=None, expending_window_size=None):

    """
    Expending window is a type of rolling window with the difference that the selected size of the window will increase by one at every step as it considers a new value. The result of this process is that the size of the window is expending and will take all the past values into account.

    # Create an expending rolling window mean feature (as defined in inputs.py):
        expending_window_size is an integer indicating the size of the window we want,
        expending_rolling_features to select the features we want to apply expending rolling window,
        X, X_train, X_test, the dataset.


    """
    for col_name in expending_features:
        X['expending_window_%s'%col_name] = X[col_name].expanding(expending_window_size).mean()
        X_train['expending_window_%s'%col_name] = X_train[col_name].expanding(expending_window_size).mean()
        X_test['expending_window_%s'%col_name] = X_test[col_name].expanding(expending_window_size).mean()
    return X, X_train, X_test
              

def clean_time_data(X, X_train, X_test, y, y_train, y_test, number_of_lags = None, window_size = None, expending_window_size = None, time_feature_name = None):
    """

    The new dataset will contain “NaN”. We should discard the first rows of the data to train the future models and drop useless columns.

    The inputs of the clean_time_data function are (as defined in inputs.py):

    number_of_lags is an integer indicating the number of lags,
    window_size is an integer indicating the size of the window,
    time_feature_name = 'date' (Name of the time feature),
    expending_window_size is an integer indicating the size of the window we want,
    X, X_train, X_test, y, y_train, y_test: the dataset processed in main.py according to inputs.py guidance

    """

    # Check the number of lines with NaNs provoked by the time series transformations and drop them from the data set
    n_lines = max([number_of_lags, window_size, expending_window_size])
    
    X = X.iloc[n_lines: , :]
    X_train = X_train.iloc[n_lines: , :]
    X_test = X_test.iloc[n_lines: , :]
    y = y.iloc[n_lines: , :].values.ravel()
    y_train = y_train.iloc[n_lines: , :].values.ravel()
    y_test = y_test.iloc[n_lines: , :].values.ravel()
        
    # Delete original time series data
    X_train = X_train.drop(columns=time_feature_name, axis = 1)
    X_test = X_test.drop(columns=time_feature_name, axis = 1)
    X = X.drop(columns=time_feature_name, axis = 1)
    
    return X, X_train, X_test, y, y_train, y_test





