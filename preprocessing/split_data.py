#!/usr/bin/python3
# split_data.py
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

# Import library
from sklearn.model_selection import train_test_split

# Split data into y (Target), X (features), training and testing.

def split_data(df, time_series, test_size = None, test_time_size = None):

    if time_series == 'yes':
        # Keep only the Target column
        y = df.loc[:, df.columns == 'Target']
        # Featues variables
        X = df.loc[:, df.columns != ('Target')]
        # Split data into train and test with test_time_size = k from inputs.py
        # It will take the last k values of the dataset for testing the model
        X_train = X[:-test_time_size]
        X_test = X[-test_time_size:]
        y_train = y[:-test_time_size]
        y_test = y[-test_time_size:]
    else:
        # We split our data to y (Target) and X (features)
        y = df.loc[:, df.columns == 'Target']
        # Features variables
        X = df.loc[:, df.columns != ('Target')]
        # Split data into train and test
        # Option test_size = 0.XX means that we take XX% of the data for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X, X_train, X_test, y, y_train, y_test,
    
    
