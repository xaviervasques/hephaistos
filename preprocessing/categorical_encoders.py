#!/usr/bin/python3
# categorical_encoders.py
# Author: Xavier Vasques (Last update: 18/01/2022)

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

# Import python libraries
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import hashlib


"""

In machine learning, we need to process different types of data. Some of them are continuous variables and others are categorical variables. In a way, we can compare the difference
between continuous and categorical data with regression and classification algorithms, at least for data inputs. As we are dealing with data, it is critically important to consider
and process the categorical data correctly to avoid any wrong impact on the performance of the machine learning models. We do not really have a choice here as we need anyway to
transform categorical data, often text, into numeric and consumable data. Most of the time, we can find three major classes of categorical data: binary, nominal, and ordinal.

It is preferable to apply continuous and categorical transformations after splitting the data between train and test. We can choose different encoders
for different features. The output, encoded data, can be merged with the rescaled continuous data to train the models. The same process is applied to the test data before applying the
trained models.

It exists different coding systems for categorical variables such as the classic encoders which are well known and widely used (ordinal, one hot, binary, frequency, hashing), the
contrast encoders that encode data by looking at different categories (or levels) of features such as Helmert or backward difference and Bayesian encoders which use the target as a
foundation for encoding. Target, leave one out, weight of evidence, James-Stein and m-estimator are Bayesian encoders. Even we already have a good list of encoders to explore, there
are many more! The important is to master a couple of them and then explore to go further.

"""

"""

We will use the categorical_encoders function in categorical_encoders.py to encode data where we invoke the encoder functions from category_encoders library or sklearn
(OrdinalEncoder, LabelEncoder, ...), specify the columns we want to encode and then call the .fit_transform() method on it with the DataFrame as the argument.

"""


def ordinal_encoding(feature_to_encode_1, X, y, X_train, y_train, X_test, y_test):

    """

    The easiest way to encode ordinal data is to assign it an integer value (integer encoding). For example, if we have a variable “size”, we can assign 0 to “small”, 1 to “medium”
    and 2 to “large”. Integer encoding is easily reversible. Ordinal encoding can be applied if there is a known relationship between categories. We can use pandas and assign the
    original order of the variable through a dictionary and then map each row for the variable as per the dictionary.

    feature_to_encode_1 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """
    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_1:
        feature_to_encode_1.remove('Target')
        X = categorical_encoders(X, y, "ordinal", feature_to_encode_1, 0, None)
        X_train = categorical_encoders(X_train,y_train, "ordinal", feature_to_encode_1, 0, None)
        X_test  = categorical_encoders(X_test, y_test, "ordinal", feature_to_encode_1, 0, None)
        y = categorical_encoders(y,y, "ordinal", ['Target'], 0, None)
        y_train = categorical_encoders(y_train,y_train, "ordinal", ['Target'], 0, None)
        y_test = categorical_encoders(y_test,y_test, "ordinal", ['Target'], 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Ordinal_Encoding')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
    else:
        X = categorical_encoders(X,y, "ordinal", feature_to_encode_1, 0, None)
        X_train = categorical_encoders(X_train,y_train, "ordinal", feature_to_encode_1, 0, None)
        X_test  = categorical_encoders(X_test, y_test, "ordinal", feature_to_encode_1, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Ordinal_Encoding')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
                
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

def one_hot_encoding(feature_to_encode_2, X, y, X_train, y_train, X_test, y_test):

    """

    One Hot Encoding is very popular. With the One Hot Encoding methodology, we will map each category to a vector containing 1 (presence) and 0 (absence).  This is applied when no
    order relationship exists. It creates new binary columns where 1 indicates the presence of each possible value from the original data.
    In this approach, for each category of a feature, we create a new column (sometimes called a dummy variable) with binary encoding (0 or 1) to denote whether a particular row
    belongs to this category. This method can be challenging if our categorical variable takes on many values and it is preferable to avoid it for variables taking more than 15
    different values.
    The drawback of this method is the size of the variable in memory since it uses as many bits as there are states meaning that the necessary memory space increases linearly with the
    number of states. Creating many columns can slow down the learning significantly.

    feature_to_encode_2 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_2:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X,y, "one_hot", feature_to_encode_2, 0, None)
        X_train = categorical_encoders(X_train,y_train, "one_hot", feature_to_encode_2, 0, None)
        X_test  = categorical_encoders(X_test,y_test, "one_hot", feature_to_encode_2, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('One_Hot_Encoding')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
    
def label_encoding(feature_to_encode_3, X, y, X_train, y_train, X_test, y_test):

    """

    With Label Encoding we replace a categorical value with a numeric value (from 0 to N with N the number of categories for the feature) to each category. If the feature contains 5
    categories, we will use 0, 1, 2, 3, and 4. This approach can bring a major issue because even if there is no relation or order between categories, the algorithm might interpret
    some order or relationship.

    feature_to_encode_3 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_3:
        feature_to_encode_3.remove('Target')
        X = categorical_encoders(X,y, "label", feature_to_encode_3, 0, None)
        X_train = categorical_encoders(X_train,y_train, "label", feature_to_encode_3, 0, None)
        X_test = categorical_encoders(X_test,y_test, "label", feature_to_encode_3, 0, None)
        y = categorical_encoders(y,y, "label", ['Target'], 0, None)
        y_train = categorical_encoders(y_train,y_train, "label", ['Target'], 0, None)
        y_test = categorical_encoders(y_test,y_test, "label", ['Target'], 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Label_Encoding')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
    else:
        X = categorical_encoders(X,y, "label", feature_to_encode_3, 0, None)
        X_train = categorical_encoders(X_train,y_train, "label", feature_to_encode_3, 0, None)
        X_test = categorical_encoders(X_test,y_test, "label", feature_to_encode_3, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Label_Encoding')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
                
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
            
def helmert_encoding(feature_to_encode_4, X, y, X_train, y_train, X_test, y_test):
    """

    Helmert Encoding compares each level of a categorical variable to the mean of the subsequent levels.

    feature_to_encode_4 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_4:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X,y, "helmert", feature_to_encode_4, 0, None)
        X_train = categorical_encoders(X_train,y_train, "helmert", feature_to_encode_4, 0, None)
        X_test = categorical_encoders(X_test,y_test, "helmert", feature_to_encode_4, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Helmert_Encoding_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)

    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

def binary_encoding(feature_to_encode_5, X, y, X_train, y_train, X_test, y_test):

    """

    The Binary Encoding method consists in different operations: the categories are encoded as ordinal, then, the resulting integers are converted into a binary code and finally the
    digits from that binary code are split into separate columns. This process results in fewer dimensions than the one hot encoding. As Helmert Encoding, we can use the
    category_encoders library to code it. We need to invoke the BinaryEncoder function by specifying the columns we want to encode and then call the .fit_transform() method on it with
    the DataFrame as the argument.

    feature_to_encode_5 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_5:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X,y, "binary", feature_to_encode_5, 0, None)
        X_train = categorical_encoders(X_train,y_train, "binary", feature_to_encode_5, 0, None)
        X_test = categorical_encoders(X_test,y_test, "binary", feature_to_encode_5, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Binary_Encoding_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data

    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
   
def frequency_encoding(feature_to_encode_6, X, y, X_train, y_train, X_test, y_test):

    """
    The Frequency Encoding method encodes by frequency which means we will create a new feature with the number of categories from the data (counts of each category).

    feature_to_encode_6 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_6:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X,y, "frequency", feature_to_encode_6, 0, None)
        X_train = categorical_encoders(X_train,y_train, "frequency", feature_to_encode_6, 0, None)
        X_test = categorical_encoders(X_test,y_test, "frequency", feature_to_encode_6, 0, None)
        filter_col_train = [col for col in X_train if col.endswith('_freq_encode')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data

    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
            
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

def mean_encoding(feature_to_encode_7, X, y, X_train, y_train, X_test, y_test):

    """
    In this method, we will encode, for each unique value of the categorical feature, based on the ratio of occurrence of the positive class in the target variable.

    feature_to_encode_7 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_7:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X,y, "mean", feature_to_encode_7, 0, None)
        X_train = categorical_encoders(X_train,y_train, "mean", feature_to_encode_7, 0, None)
        X_test = categorical_encoders(X_test,y_test, "mean", feature_to_encode_7, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Mean_Encoding_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data

    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)

    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

def sum_encoding(feature_to_encode_8, X, y, X_train, y_train, X_test, y_test):

    """
    Sum encoding method, also called effect of deviation encoding, will compare the mean of the target (dependent variable) for a given level of a categorical column to the overall
    mean of the target. It’s like One Hot Encoding with the difference that we use 1, 0 and -1 values to encode the data. It can be used in Linear Regression types of models. It can
    be coded with the category_encoders library.

    feature_to_encode_8 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    
    if 'Target' in feature_to_encode_8:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "sum", feature_to_encode_8, 0, None)
        X_train = categorical_encoders(X_train, y_train, "sum", feature_to_encode_8, 0, None)
        X_test = categorical_encoders(X_test, y_test, "sum", feature_to_encode_8, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Sum_Encoding_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data

    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)

    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
    
def weightofevidence_encoding(feature_to_encode_9, X, y, X_train, y_train, X_test, y_test):

    """

    The Weight of Evidence (WoE) is coming from the credit scoring world and measures the “strength” of a grouping technique to separate the good customers and bad customers which
    refers to the customers who defaulted on a loan or not. In the context of machine learning, WoE is also used for the replacement of categorical values. With One Hot Encoding, if
    we assume that a column contains 5 unique labels, there will be 5 new columns. Here, we will replace the values by the WoE. This method is particularly well suited for subsequent
    modeling using Logistic Regression. WoE transformation orders the categories on a “logistic” scale which is natural for logistic regression.

    feature_to_encode_9 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_9:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "weightofevidence", feature_to_encode_9, 0, None)
        X_train = categorical_encoders(X_train, y_train, "weightofevidence", feature_to_encode_9, 0, None)
        X_train = categorical_encoders(X_train, y_train, "weightofevidence", feature_to_encode_9, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('WoE_Encoding_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)

    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

def probability_ratio_encoding(feature_to_encode_10, X, y, X_train, y_train, X_test, y_test):

    """

    Probability Ratio Encoding is similar to WoE but we will only keep the ratio, not the logarithm of it. For each category, the mean of the target is calculated to equal 1 that is
    the probability p(1) of being 1 and the probability p(0) of not being 1 (it’s 0). The ratio of happening and not happening is simply p(1)/p(0). All the categorical values should be
    replaced with this ratio.

    feature_to_encode_10 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_10:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "probabilityratio", feature_to_encode_10, 0, None)
        X_train = categorical_encoders(X_train, y_train, "probabilityratio", feature_to_encode_10, 0, None)
        X_test = categorical_encoders(X_test, y_test, "probabilityratio", feature_to_encode_10, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Proba_Ratio_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

def hashing_encoding(feature_to_encode_11, X, y, X_train, y_train, X_test, y_test):

    """

    Hashing encoding is similar to One-Hot-encoding which converts the category into binary numbers using new variables. The difference is that we can fix the number of variables we
    want. Hashing encoding maps each category to an integer within a pre-defined range with the help of the hash function. We can use different hashing methods using the hash_method
    option. Any method from hashlib works (import hashlib) -- this is defined in inputs.py (hash_method). We also need to choose the number of components (n_components in inputs.py).
    If we want 4 binary features, we can convert the output written in binary and select the last 4 bits.

    feature_to_encode_11 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_11:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "hashing", feature_to_encode_11, inputs.n_components, inputs.hash_method)
        X_train = categorical_encoders(X_train, y_train, "hashing", feature_to_encode_11, inouts.n_components, inputs.hash_method)
        X_test = categorical_encoders(X_test, y_test, "hashing", feature_to_encode_11, inputs.n_components, inputs.hash_method)
        filter_col_train = [col for col in X_train if col.startswith('Hashing_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
    
def backward_difference_encoding(feature_to_encode_12, X, y, X_train, y_train, X_test, y_test):

    """

    In backward difference encoding method, which is similar to Helmert encoding, the mean of the dependent variable for a level is compared with the mean of the dependent variable for
    the prior level. Backward difference encoding falls under the contrast encoders for categorical features. Backward difference encoding may be useful for both nominal and ordinal
    variable. In addition, contrary to the dummy encoding examples, we will see as outputs regressed continuous values.

    feature_to_encode_12 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """
    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_12:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "backwarddifference", feature_to_encode_12, 0, None)
        X_train = categorical_encoders(X_train, y_train, "backwarddifference", feature_to_encode_12, 0, None)
        X_test = categorical_encoders(X_test, y_train, "backwarddifference", feature_to_encode_12, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Backward_Diff_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)

    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
    
def leave_one_out_encoding(feature_to_encode_13, X, y, X_train, y_train, X_test, y_test):

    """

    The target-based encoder Leave One Out encoding excludes the current row’s target when we calculate the mean target for a level to reduce the effect of outliers. In other words, it
    involves taking the mean target value of all data points in the category except the current row.

    feature_to_encode_13 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_13:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "leaveoneout", feature_to_encode_13, 0, None)
        X_train = categorical_encoders(X_train, y_train, "leaveoneout", feature_to_encode_13, 0, None)
        X_test = categorical_encoders(X_test, y_test, "leaveoneout", feature_to_encode_13, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('Leave_One_Out_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
          
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
          
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)

    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
    
def james_stein_encoding(feature_to_encode_14, X, y, X_train, y_train, X_test, y_test):

    """

    The target-based encoder James-Stein, only defined for normal distributions, is inspired by James-Stein estimator.
    For the feature value i, James-Stein estimator return a weighted average.

    feature_to_encode_14 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_14:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "jamesstein", feature_to_encode_14, 0, None)
        X_train = categorical_encoders(X_train, y_train, "jamesstein", feature_to_encode_14, 0, None)
        X_test = categorical_encoders(X_test, y_test, "jamesstein", feature_to_encode_14, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('James_Stein_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num
    
def m_estimator_encoding(feature_to_encode_15, X, y, X_train, y_train, X_test, y_test):

    """

    M-estimator encoding, a more general Bayesian approach, has only one hyperparameter (m) which represents the power of regularization and generally good for high cardinality data.
    The default value of m is 1. The recommended values are in the range of 1 to 100 and higher is m stronger shrinking.

    feature_to_encode_15 are the features selected in inputs.py to encode
    X, y, X_train, y_train, X_test, y_test: the dataset

    """

    # Variable to store selected categorical features allowing to split continuous (df_num) and categorical (df_cat) data
    categorical_features = []
    if 'Target' in feature_to_encode_15:
        print("Do another categorical encoding method for the variable: Target")
        exit()
    else:
        X = categorical_encoders(X, y, "mestimator", feature_to_encode_15, 0, None)
        X_train = categorical_encoders(X_train, y_train, "mestimator", feature_to_encode_15, 0, None)
        X_test = categorical_encoders(X_test, y_test, "mestimator", feature_to_encode_15, 0, None)
        filter_col_train = [col for col in X_train if col.startswith('M_Estimator_')]
        categorical_features = categorical_features + filter_col_train # we can be into a situation where you can’t encode a categorical test predictor because it didn’t appear in the training data or vice versa. Test data should be representative of training data
           
    X_cat = pd.DataFrame(X, columns = categorical_features)
    X_num = X.drop(columns=categorical_features, axis = 1)
    X_train_cat = pd.DataFrame(X_train, columns = categorical_features)
    X_train_num = X_train.drop(columns=categorical_features, axis = 1)
    X_test_cat = pd.DataFrame(X_test, columns = categorical_features)
    X_test_num = X_test.drop(columns=categorical_features, axis = 1)
    
    return X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num

# Categorical encoders functions
def categorical_encoders(df, y, encoder, feature_to_encode, n_components, hash_method):

    if encoder == "ordinal":
        try:
            df[feature_to_encode] = ordinalencoding(df[feature_to_encode])
        except:
            print("Something went wrong with Ordinal Encoder: Please Check")
            exit()
        
    if encoder == "one_hot":
        try:
            df = pd.get_dummies(df, prefix="One_Hot", columns=feature_to_encode)
        except:
            print("Something went wrong with One Hot Encoder: Please Check")
            exit()
        
    if encoder == "label":
        try:
            df[feature_to_encode] = labelencoding(df[feature_to_encode])
        except:
            print("Something went wrong with Label Encoder: Please Check")
            exit()
            
    if encoder == "helmert":
        try:
            Y = helmertencoding(df[feature_to_encode])
            df = df.drop(feature_to_encode, axis = 1)
            df = pd.concat([df, Y], axis=1)
        except:
            print("Something went wrong with Helmert Encoder: Please Check")
            exit()
            
    if encoder == "binary":
        try:
            Y = binaryencoding(df[feature_to_encode])
            df = df.drop(feature_to_encode, axis = 1)
            df = pd.concat([df, Y], axis=1)
        except:
            print("Something went wrong with Binary Encoder: Please Check")
            exit()
            
    if encoder == "frequency":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # grouping by frequency
                frequency = df.groupby(col_name).size()/len(df)
                # mapping values to dataframe
                df.loc[:,"{}_freq_encode".format(col_name)] = df[(col_name)].map(frequency)
                # drop original column
                df = df.drop([col_name], axis = 1)
        except:
            print("Something went wrong with Frequency Encoder: Please Check")
            exit()
    
    
    if encoder == "mean":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of TargetEncoder
                mean_encoder = ce.TargetEncoder(drop_invariant=True)
                # Assigning numerical value and storing it
                df_encoded = mean_encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('Mean_Encoding_')
                # Concatenate dataframe and drop original column
                df = pd.concat([df, df_encoded], axis=1)
                df = df.drop([col_name], axis = 1)
        except:
            print("Something went wrong with Mean Encoder: Please Check")
            exit()
                   
    if encoder == "sum":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of SumEncoder
                sum_encoder = ce.SumEncoder(drop_invariant=True)
                # Assigning numerical value and storing it
                df_encoded = sum_encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('Sum_Encoding_')
                # Concatenate dataframe and drop original column
                df = pd.concat([df, df_encoded], axis=1)
                df = df.drop([col_name], axis = 1)
        except:
            print("Something went wrong with Sum Encoder: Please Check")
            exit()
            
    if encoder == "weightofevidence":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of SumEncoder
                #regularization is mostly to prevent division by zero.
                woe = ce.WOEEncoder(random_state=42, regularization=0)
                # Assigning numerical value and storing it
                df_encoded = woe.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('WoE_Encoding_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
                print("Something went wrong with Weight of Evidence Encoder: Please Check")
                exit()
            

    if encoder == "probabilityratio":
        try:
            df = pd.concat([y, df],axis=1)
            for col_name in feature_to_encode:
                # Calculation of the probability of target being 1
                probability_encoding_1 = df.groupby(col_name)['Target'].mean()
                print(probability_encoding_1)
                # Calculation of the probability of target not being 1
                probability_encoding_0 = 1 - probability_encoding_1
                probability_encoding_0 = np.where(probability_encoding_0 == 0, 0.00001, probability_encoding_0)
                # Probability ratio calculation
                df_encoded = probability_encoding_1 / probability_encoding_0
                # Map the probability ratio into the data
                df.loc[:,'Proba_Ratio_%s'%col_name] = df[col_name].map(df_encoded)
                # Drop feature to let the transformed one
                df = df.drop([col_name], axis = 1)
            df = df.drop(['Target'], axis = 1)
        except:
            print("Something went wrong with Probability Ratio Encoder: Please Check")
            exit()
    
    if encoder == "hashing":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of HashingEncoder
                # n_components contains the number of bits you want in your hash value.
                encoder_purpose = ce.HashingEncoder(n_components=n_components, hash_method=hash_method)
                # Assigning numerical value and storing it
                df_encoded = encoder_purpose.fit_transform(df[col_name])
                # We renanme columns to identify which feature we transformed
                for x in range(n_components):
                    df_encoded = df_encoded.rename(columns={"col_%i"%x: "%s_%s_%i"%('Hashing',col_name,x)})
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with Hashing Encoder: Please Check")
            exit()
            
    if encoder == "backwarddifference":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of BackwardDifferenceEncoder
                encoder = ce.BackwardDifferenceEncoder(cols=col_name,drop_invariant=True)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name])
                df_encoded = df_encoded.add_prefix('Backward_Diff_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with Backward Difference Encoder: Please Check")
            exit()

    if encoder == "leaveoneout":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of BackwardDifferenceEncoder
                encoder = ce.LeaveOneOutEncoder(cols=col_name)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('Leave_One_Out_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with Leave One Out Encoder: Please Check")
            exit()
            
            
    if encoder == "jamesstein":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of BackwardDifferenceEncoder
                encoder = ce.JamesSteinEncoder(cols=col_name)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('James_Stein_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with James-Stein Encoder: Please Check")
            exit()
    
    if encoder == "mestimator":
        try:
            # Get all features
            for col_name in feature_to_encode:
                # Creating an instance of MEstimateEncoder
                encoder = ce.MEstimateEncoder(cols=col_name)
                # Assigning numerical value and storing it
                df_encoded = encoder.fit_transform(df[col_name], y)
                df_encoded = df_encoded.add_prefix('M_Estimator_')
                # Drop original column and concatenate dataframe
                df = df.drop([col_name], axis = 1)
                df = pd.concat([df, df_encoded], axis=1)
        except:
            print("Something went wrong with M-Estimator Encoder: Please Check")
            exit()
            
    return(df)
               
def ordinalencoding(X):
    print("test")
    print(X)
    # Creating an instance of Ordinalencoder
    enc = OrdinalEncoder()
    # Assigning numerical value and storing it
    enc.fit(X)
    X = enc.transform(X)
    X = X.add_prefix('Ordinal_Encoding_')
    return X
    
def onehotencoding(X):
    X = pd.get_dummies(X, prefix="One_Hot_Encoding")
    return X
    
def labelencoding(X):
    # Creating an instance of Labelencoder
    enc = LabelEncoder()
    # Assigning numerical value and storing it
    X = X.apply(enc.fit_transform)
    X = X.add_prefix('Label_Encoding_')
    return X
        
def helmertencoding(X):
    # Creating an instance of HelmertEncoder
    enc = ce.HelmertEncoder(drop_invariant=True)
    # Assigning numerical value and storing it
    X = enc.fit_transform(X)
    X = X.add_prefix('Helmert_Encoding_')
    return X
            
def binaryencoding(X):
    # Creating an instance of BinaryEncoder
    enc = ce.BinaryEncoder()
    # Assigning numerical value and storing it
    df_binary = enc.fit_transform(X)
    df_binary = df_binary.add_prefix('Binary_Encoding_')
    return df_binary


