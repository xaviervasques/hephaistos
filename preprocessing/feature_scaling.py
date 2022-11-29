#!/usr/bin/python3
# preprocessing.py
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

# Import necessary python libraries
import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

"""

Most of the time, we encounter different types of variables with different ranges in the same dataset which can differ a lot.  If we use the data with the original scale, we will
certainly put more weight on the variables with a large range. Therefore, we need to apply what we call features rescaling to make sure the variables are almost on the same scale. It
allows comparing features (apple to apples) as equally important.

"""


def standardscaler(data):

    """

    The result of the dataset standardization (or Z-score normalization) is that the features will be rescaled to ensure the mean and the standard deviation to be 0 and 1,
    respectively.
    It is necessary when continuous independent variables are measured at different scales such as in different measurement units.

    """
    
    try:
        # define StandardScaler
        StandardScaler = preprocessing.StandardScaler()
        # transform data
        scale = StandardScaler.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with StandardScaler: Check your data")
        
def minmaxscaler(data):

    """

    We also can transform features by scaling each of them to a defined range (e.g., between -1 and 1 or 0 and 1). Min-max scaling (MinMaxScaler) can for instance be very useful for
    some machine learning models. MinMax scaling has some advantages over StandardScaler when data distribution is not Gaussian and the feature falls within a bounded interval which is
    typically the case on pixel intensity’s fitting within a 0-255 range.

    """

    try:
        # define MinMaxScaler
        min_max_scaler = preprocessing.MinMaxScaler()
        # transform data
        scale = min_max_scaler.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with MinMaxScaler: Check your data")
        
def maxabsscaler(data):

    """

    The MaxAbsScaler is like the MinMaxScaler with the difference that it automatically scales the data between 0 and 1 based on the absolute maximum. This scaler is specifically
    suitable for data that is already centered at zero or sparse data and does not center the data which keeps sparsity.

    """

    try:
        # define MaxAbsScaler
        max_abs_scaler = preprocessing.MaxAbsScaler()
        # transform data
        scale = max_abs_scaler.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with MaxAbsScaler: Check your data")
        
def robustscaler(data):

    """

    If your data contains an important number of outliers, the use of the mean and variance to scale the data will probably not work correctly. In this case, an option is the use of
    RobustScaler which removes the median and scales the data according to the quantile range.

    """

    try:
        # define RobustScaler
        RobustScaler = preprocessing.RobustScaler()
        # transform data
        scale = RobustScaler.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with RobustScaler: Check your data")
        
def normalize(data):

    """

    Normalizer is also a technique of scaling individual samples to have unit norm which is a common operation for clustering or text classification. We need to normalize data when our
    model predicts based on the weighted relationship formed between data points. To give a mental image, standardization is a column-wise operation while normalization is a row-wise
    operation. As standardization, we have different ways to normalize: l1, l2, max.

    By default, in scikit-learn, Normalizer uses l2. We can change it using the norm option (‘l1’, ‘l2’, ‘max’).

    """

    try:
        # define Normalizer
        Normalize = preprocessing.Normalizer()
        # transform data
        scale = Normalize.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Normalizer: Check your data")
        
def logtransformation(data):

    """

    The lognormal transformation converts the values to a lognormal scale
    In log transformation, each variable of x will be replaced by log (x) with natural, base 10 or base 2 log

    """

    try:
        # define log transformation
        log_target = np.log1p(data)
        data_scaling = pd.DataFrame(log_target, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Log Tranformation: Check your data")
        
def squareroottransformation(data):

    """

    In square root transformation, x will be replaced by the square root(x). It will give moderate effect but can be applied to zero values.

    """

    try:
        # define square root transformation and transform the data
        sqrrt_target = data**(1/2)
        data_scaling = pd.DataFrame(sqrrt_target, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Square Root Tranformation: Check your data")
        
def reciprocaltransformation(data):

    """

    In reciprocal transformation, x will be replaced by its inverse (1/x).

    """

    try:
        # tranforme the data
        re_target = 1/data
        data_scaling = pd.DataFrame(re_target, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Reciprocal Transformation: Check your data")
    
    
def boxcoxtransformation(data):

    """

    Box-Cox transformation.

    """

    try:
        # define and transform the data
        pt = PowerTransformer(method="box-cox")
        scale = pt.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Box-Cox: Check your data")

def yeojohnsontransformation(data):

    """

    Yeo-Johnson transformation.

    """

    try:
        # define and transform the data
        pt = PowerTransformer(method="yeo-johnson")
        scale = pt.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Yeo-Johnson: Check your data")
        
        
def quantiletransformationgaussian(data):

    """

    The nonparametric quantile transformation transforms the data to a certain data distribution such as normal distribution by applying quantile function, an inverse function
    of the cumulative distribution function (CDF), into the data.

    """

    try:
        # define QuantileTransformer
        qtg = QuantileTransformer(n_quantiles=1000, output_distribution="normal")
        # transform the data
        scale = qtg.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Quantile Transformation with Gaussian Distribution: Check your data")

def quantiletransformationuniform(data):

    """

    The nonparametric quantile transformation transforms the data to a certain data distribution such as uniform distribution by applying quantile function, an inverse function
    of the cumulative distribution function (CDF), into the data.

    """

    try:
        qtg = QuantileTransformer(n_quantiles=1000, output_distribution="uniform")
        scale = qtg.fit_transform(data)
        data_scaling = pd.DataFrame(scale, columns = data.columns)
        return data_scaling
    except:
        print("Something went wrong with Quantile Transformation with Uniform Distribution: Check your data")
