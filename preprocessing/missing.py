#!/usr/bin/python3
# missing.py
# Author: Xavier Vasques (Last update: 10/04/2022)

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


# Importing all packages required for imputation
import sys
import numpy as np
import pandas as pd


"""

The issue of missing data in machine learning has been largely overlooked as it affects data analysis across a wide range of domains. The handling of missing values is also a key step
in the preprocessing of the dataset as many machine learning algorithms do not support missing values. In addition, making the right decision can generate robust data models. Missing
values are due to different reasons such as incomplete extraction, data corruption, or some issues when loading the dataset.

There are a large set of methodologies to handle missing values ranging from simple ones such as deleting the rows containing missing values, impute them for both continuous and
categorical variables to more complex such as the use of machine and deep learning algorithms to impute missing values.

"""

# using row removal
def row_removal(df):
    original_row, original_col = df.shape[0], df.shape[1]
    print('Using row removal algorithm...')
    # removing rows
    df_row = df.dropna(axis=0)
    print(f"Shape of new dataframe : {df_row.shape}")
    print(f"Total {original_row - df_row.shape[0]} rows removed")
    return df_row

# using column removal
def column_removal(df):
    original_row, original_col = df.shape[0], df.shape[1]
    print('Using column removal algorithm...')
    print('Warning : Features may be reduced, introducing inconsistency when Testing !')
    # removing columns
    df_col = df.dropna(axis=1)
    print(f"Shape of new dataframe : {df_col.shape}")
    print(f"Total {original_col - df_col.shape[1]} columns removed")
    return df_col

# function for detecting missing values and reporting it
def detect_missing(df):
    # checking missing values
    null_series = df.isnull().sum()
    null_column_list = []
    if sum(null_series):
        print('Following columns contains missing values : ')
        total_samples = df.shape[0]
        for i, j in null_series.items():
            if j:
                print("{} : {:.2f} %".format(i, (j/total_samples)*100))
                null_column_list.append(i)
    else:
        print("None of the columns contains missing values !")
    return null_column_list

# using statistical imputation (Mean for continuous variables and Mode for categorical variables)
def stats_imputation_mean(df):

    """
    Replacing missing values by the mean, median or mode can be applied on features which have numeric data. The mean imputation will replace any missing value from a variable
    with the mean of that variable for all other cases which has the great advantage to not change the sample mean of the variable. The drawback of the mean is that it will
    attenuate any correlation involving the imputed variables. Mode is frequently used to impute missing values. It works with categorical features by replacing the missing value
    by most frequent category within the variable.
    This is obviously approximations but can improve model performance by avoiding data removal. There are different possibilities to optimize the results such as using the
    deviation of neighboring values when the data is linear.
    """
    null_column_list = detect_missing(df)

    print('Using Statistical imputation algorithm...')
    # extracting columns for numerical columns
    valid_cols = [column for column in null_column_list if df[column].dtype != 'object']
    # extracting columns for categorical columns
    categorical_cols = [column for column in null_column_list if df[column].dtype == 'object']
    numeric_cols = valid_cols
    df_stats_mean, df_stats_mode = df.copy(), df.copy()
    # Imputing mean for numeric values and then imputing median and mode for categorical values
    print(f'Imputing following columns with mean : {numeric_cols}')
    print(f'Imputing following columns with mode : {categorical_cols}')
    if len(numeric_cols):
        for i in numeric_cols:
            df_stats_mean.fillna({i : df[i].mean()}, inplace=True)
            
    if len(categorical_cols):
        for i in categorical_cols:
            df_stats_mode.fillna({i : df[i].mode()[0]}, inplace=True)

    return df_stats_mean, df_stats_mode

# using statistical imputation (Median for continuous variables and Mode for categorical variables)
def stats_imputation_median(df):
    null_column_list = detect_missing(df)
    print('Using Statistical imputation algorithm...')
    # extracting columns for numerical columns
    valid_cols = [column for column in null_column_list if df[column].dtype != 'object']
    # extracting columns for categorical columns
    categorical_cols = [column for column in null_column_list if df[column].dtype == 'object']
    numeric_cols = valid_cols
    df_stats_median, df_stats_mode = df.copy(), df.copy()
    # Imputing mean for numeric values and then imputing median and mode for categorical values
    print(f'Imputing following columns with median: {numeric_cols}')
    print(f'Imputing following columns with mode : {categorical_cols}')
    if len(numeric_cols):
        for i in numeric_cols:
            df_stats_median.fillna({i : df[i].median()}, inplace=True)
            
    if len(categorical_cols):
        for i in categorical_cols:
            df_stats_mode.fillna({i : df[i].mode()[0]}, inplace=True)

    return df_stats_median, df_stats_mode

# using statistical imputation with mode
def stats_imputation_mode(df):
    null_column_list = detect_missing(df)
    print('Using Statistical imputation algorithm...')
    # extracting columns for numerical columns
    valid_cols = [column for column in null_column_list if df[column].dtype != 'object']
    # extracting columns for categorical columns
    categorical_cols = [column for column in null_column_list if df[column].dtype == 'object']
    numeric_cols = valid_cols
    df_stats_mode = self.df.copy()
    # Imputing mean for numeric values and then imputing median and mode for categorical values
    print(f'Imputing following columns with mode : {numeric_cols}')
    print(f'Imputing following columns with mode : {categorical_cols}')
    if len(numeric_cols):
        for i in numeric_cols:
            df_stats_mode.fillna({i : df[i].mode()[0]}, inplace=True)

    if len(categorical_cols):
        for i in categorical_cols:
            df_stats_mode.fillna({i : df[i].mode()[0]}, inplace=True)

    return df_stats_mode

# using linear interpolation
def linear_interpolation(df):

    """
    Interpolation works well for a time series with some trend but not suitable for seasonal data. It tries to estimate values from other observations within the range of a
    discrete set of known data points. In other words, it will adjust a function to our data and uses this function to extrapolate the missing data. The simplest type of
    interpolation is the linear interpolation which is the method of approximating value by joining dots in increasing order along a straight line. It makes a mean between the
    values before the missing data and the value after.
    
    """
    
    null_column_list = detect_missing(df)
    print('Using Linear Interpolation imputation algorithm...')
    # extracting columns for numerical columns
    valid_cols = [column for column in null_column_list if df[column].dtype != 'object']
    # extracting columns for categorical columns
    categorical_cols = [column for column in null_column_list if df[column].dtype == 'object']
    numeric_cols = valid_cols
           
    df_linear_interpolation = df.copy()
    # Linear interpolation for numeric values
    print(f'Imputing following columns with linear interpolation : {numeric_cols}')

    if len(numeric_cols):
        for i in numeric_cols:
            df_linear_interpolation[numeric_cols] = df_linear_interpolation[numeric_cols].interpolate(method='linear', limit_direction='forward', axis=0)

    if len(categorical_cols):
        for i in categorical_cols:
            df_linear_interpolation[numeric_cols] = df_linear_interpolation[numeric_cols].interpolate(method='linear', limit_direction='forward', axis=0)

    return df_linear_interpolation

# using MICE
def mice(df):

    """
    
    In multiple imputation, several different completed set of data are generated and each missing value is replaced by several different values. The procedure has different steps:
        - The dataset with missing values is duplicated several times,
        - The missing values of each duplicate are replaced with imputed values. Due to random variation across copies, slightly different values are imputed,
        - The multiple imputed datasets are each analyzed, and the results are combined.
    The advantage of multiple imputations, as opposed to single imputations, accounts for the statistical uncertainty in the imputations.
    
    """

    print('Using MICE imputation algorithm...')
    from fancyimpute import IterativeImputer
    MICE_imputer = IterativeImputer()
    df_mice = df.copy(deep=True)
    df_mice.iloc[:, :] = MICE_imputer.fit_transform(df_mice)
    return df_mice
    
# using KNN
def knn(df):

    """
    The KNN Imputer method uses the K-Nearest Neighbors algorithm to replace the missing values by identifying the neighboring points through a measure of distance, by default the
    Euclidean distance, and the missing values can be estimated using completed values of neighboring observations. We can use for example the mean value from the nearest k neighbors
    (n_neighbors) in the dataset. It can be used for continuous, discrete, and categorical data.
    
    """

    print('Using KNN imputation algorithm...')
    from sklearn.impute import KNNImputer
    KNN_imputer = KNNImputer(n_neighbors=5)
    df_knn = df.copy(deep=True)
    df_knn.iloc[:, :] = KNN_imputer.fit_transform(df_knn)
    return df_knn

