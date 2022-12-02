#!/usr/bin/python3
# ml_pipeline_function.py
# Author: Xavier Vasques (Last update: 29/05/2022)

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

'''

ml_pipeline_function.py takes a dataframe (df) as input and processes the choosen methods (Features Engineering, Classification, Regression ...)

'''

# Import python libraries
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.linear_model import LinearRegression

# Import hephaestos libraries
import preprocessing.feature_scaling
import preprocessing.split_data
import preprocessing.time_series
import preprocessing.categorical_encoders
import preprocessing.feature_extraction
import preprocessing.feature_selection
import preprocessing.missing
import classification.classification_cpu
import classification.classification_gpu
import classification.classification_qpu

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from preprocessing.missing import row_removal, column_removal, stats_imputation_mean, stats_imputation_median, stats_imputation_mode, linear_interpolation, mice, knn
from preprocessing.feature_scaling import standardscaler, normalize, minmaxscaler, maxabsscaler, robustscaler, logtransformation, squareroottransformation, reciprocaltransformation, boxcoxtransformation, yeojohnsontransformation, quantiletransformationgaussian, quantiletransformationuniform
from preprocessing.feature_extraction import pca, ica, icawithpca, lda_extraction, random_projection, truncatedSVD, isomap, standard_lle, modified_lle, hessian_lle, ltsa_lle, mds, spectral, tsne, nca
from preprocessing.split_data import split_data
from preprocessing.feature_selection import variance_threshold, chi_square, anova_f_c, anova_f_r, pearson, forward_stepwise, backward_elimination, exhaustive, lasso, feat_reg_ml, embedded_linear_regression, embedded_logistic_regression, embedded_decision_tree_regressor, embedded_decision_tree_classifier, embedded_random_forest_regressor, embedded_random_forest_classifier, embedded_permutation_regression, embedded_permutation_classification, embedded_xgboost_regression, embedded_xgboost_classification
from preprocessing.categorical_encoders import ordinal_encoding, one_hot_encoding, label_encoding, helmert_encoding, binary_encoding, frequency_encoding, mean_encoding, sum_encoding, weightofevidence_encoding, probability_ratio_encoding, hashing_encoding, backward_difference_encoding, leave_one_out_encoding, james_stein_encoding, m_estimator_encoding
from classification.classification_cpu import svm_linear, svm_rbf, svm_sigmoid, svm_poly, logistic_regression, lda, qda, gnb, mnb, kneighbors, sgd, nearest_centroid, decision_tree, random_forest, extra_trees, mlp_neural_network_auto, mlp_neural_network
from classification.classification_gpu import gpu_logistic_regression, gpu_mlp, gpu_rnn, conv2d
from regression.regression_cpu import linear_regression, svr_linear, svr_rbf, svr_sigmoid, svr_poly, mlp_regression, mlp_auto_regression
from regression.regression_gpu import gpu_linear_regression, gpu_mlp_regression, gpu_rnn_regression
from classification.classification_qpu import q_kernel_default, q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_zz, q_kernel_default_pegasos, q_kernel_8_pegasos, q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos, q_kernel_zz_pegasos, q_kernel_training, q_twolayerqnn, q_circuitqnn, q_vqc


def ml_pipeline_function(df=None, X=None, y=None, X_train=None, y_train=None, X_test=None, y_test=None, output_folder = None, missing_method = None, test_size = None, test_time_size = None, time_split = None, time_feature_name = None, time_format = None, time_transformation = None, lagged_features = None, number_of_lags = None, lag_aggregation = None, rolling_features = None, window_size = None, expending_features = None, expending_window_size = None, categorical = None, features_ordinal = None, features_one_hot = None, features_label = None, features_helmert = None, features_binary = None, features_frequency = None, features_mean = None, features_sum = None, features_weight = None, features_proba_ratio = None, features_hashing = None, features_backward = None, features_leave_one_out = None, features_james_stein = None, features_m = None, rescaling = None, features_extraction = None, number_components = None, n_neighbors = None, feature_selection = None, k_features = None, features_to_variance = None, var_threshold = None, cc_features = None, cc_target = None, wrapper_classifier = None, min_features = None, max_features = None, lasso_alpha = None, ml_penalty = None, classification_algorithms = None, regression_algorithms = None, n_estimators_forest = None, n_estimators_extra = None, cv = None, activation = None, optimizer = None, epochs = None, learning_rate = None, quantum_algorithms = None, feature_dimension = None, reps = None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, max_iter = None, hidden_layer_sizes = None, mlp_activation = None, solver = None, alpha = None, mlp_learning_rate = None, learning_rate_init = None, gpu_logistic_optimizer = None, gpu_logistic_epochs = None, gpu_logistic_loss = None, gpu_mlp_optimizer = None, gpu_mlp_epochs= None, gpu_mlp_loss= None, gpu_mlp_activation = None, max_iter_r = None, hidden_layer_sizes_r = None, mlp_activation_r = None, solver_r = None, alpha_r = None, mlp_learning_rate_r = None, learning_rate_init_r = None, gpu_mlp_epochs_r = None, gpu_mlp_activation_r = None, rnn_units = None, rnn_activation = None, rnn_optimizer = None, rnn_loss = None, rnn_epochs = None, convolutional = None, conv_activation=None, conv_kernel_size=None, conv_optimizer=None, conv_loss=None, conv_epochs=None, multiclass = None):

    """
    We will apply all the desired procedures to the dataframe df
    
    """
    
    if(missing_method is not None):
    # See missing.py to read explanations regarding the functions, inputs and outputs.
        print("\n")
        print("############################ Handling Missing Values: START ")
        print("\n")
    
        if missing_method == 'row_removal':
            df = row_removal(df)
        if missing_method == 'column_removal':
            df = column_removal(df)
        if missing_method == 'stats_imputation_mean':
            df = stats_imputation_mean(df)
        if missing_method == 'stats_imputation_median':
            df = stats_imputation_median(df)
        if missing_method == 'stats_imputation_mode':
            df = stats_imputation_mode(df)
        if missing_method == 'linear_interpolation':
            df = linear_interpolation(df)
        if missing_method == 'mice':
            df = mice(df)
        if missing_method == 'knn':
            df = knn(df)
        
        print("\n")
        print("Printing data after handling missing values")
        print("\n")
        print(df)
        print("\n")
        print("############################ Handling Missing Values: END ")
        print("\n")

    else:
        print("No missing values method selected")
        missing_method = 'None'
        print("\n")
            
    print("\n")
    print("############################ Time Series Transformation: START ")
    print("\n")
            
    if(test_time_size is not None):
            print("Time series data\n")
            time_series = 'yes'
            test_size = None
            # Split data into y (Target), X (Features)
            print("\n")
            print("Splitting data to X, X_train, X_test, y, y_train, y_test  ...")
            print("\n")
            X, X_train, X_test, y, y_train, y_test = split_data(df, time_series, test_size, test_time_size)
            
            # Split the time information by day, month, year, hours, seconds, milliseconds, etc
            from preprocessing.time_series import split
            X, X_train, X_test = split(time_split, time_feature_name, time_format, X, X_train, X_test)
                                                
            # Adding lags to the data
            if time_transformation == 'lag':
                print("LAG SELECTED")
                from preprocessing.time_series import lag
                X, X_train, X_test, y, y_train, y_test = lag(X, y, X_train, y_train, X_test, y_test, lagged_features, number_of_lags, lag_aggregation)
            else:
                number_of_lags = 0
            
            # Create a rolling window mean feature
            if time_transformation =='rolling_window':
                from preprocessing.time_series import rolling_window
                X, X_train, X_test = rolling_window(X, X_train, X_test, rolling_features, window_size)
            else:
                window_size = 0
                
            
            # Create an expending rolling window mean feature
            if time_transformation == 'expending_window':
                from preprocessing.time_series import expending_window
                X, X_train, X_test = expending_window(X, X_train, X_test, expending_features, expending_window_size)
            else:
                expending_window_size = 0
            
            # Clean data after time series transformation (NaNs) and delete original time series data
            from preprocessing.time_series import clean_time_data
            print(time_feature_name)
            X, X_train, X_test, y, y_train, y_test = clean_time_data(X, X_train, X_test, y, y_train, y_test, number_of_lags, window_size, expending_window_size, time_feature_name)
        
            print("\n")
            print("Printing X following time series transformation: ")
            print("\n")
            print(X)
            print("\n")
            print(y)
    
            print("\n")
            print("Printing X_train and y_train following time series transformation: ")
            print("\n")
            print(X_train)
            print("\n")
            print(y_train)
    
            print("\n")
            print("Printing X_test and y_test following time series transformation: ")
            print("\n")
            print(X_test)
            print("\n")
            print(y_test)
            print("\n")
        
            print("\n")
            print("############################ Time Series Transformation: END ")
            print("\n")
    
    else:
        if convolutional is not None:
            if 'conv2d' in convolutional:
                print("\n")
                print("Splitting Data for Convolutional Neural Networks (conv2d) + One Hot Encoding y, y_train, y_test")
                print("\n")
                
                X = X
                y = y
                X_train = X_train
                X_test = X_test
                # One-hot encode target column
                # A column will be created for each output category.
                # For example, for an image with the number 2 we will have [0,1,0,0,0,0,0,0,0,0]
                y_train = to_categorical(y_train)
                y_test = to_categorical(y_test)
                y = to_categorical(y)

        else:
                                
            print("\n")
            print("No Time Series Transformation Selected\n")
            print("\n")
            print("############################ Time Series Transformation: END ")
            print("\n")
                
            time_series, test_time_size = None, None
            # Split data into y (Target), X (Features)
            print("\n")
            print("Splitting data to X, X_train, X_test, y, y_train, y_test  ...")
            print("\n")
            X, X_train, X_test, y, y_train, y_test = split_data(df, time_series, test_size, test_time_size)
    
        print("\n")
        print("Printing original data X and y: ")
        print("\n")
        print(X)
        print("\n")
        print(y)
    
        print("\n")
        print("Printing X_train and y_train: ")
        print("\n")
        print(X_train)
        print("\n")
        print(y_train)
    
        print("\n")
        print("Printing X_test and y_test: ")
        print("\n")
        print(X_test)
        print("\n")
        print(y_test)
        print("\n")

    if(categorical is not None):
        # See categorical_encoders.py to read explanations regarding the functions, inputs and outputs.
    
        print("\n")
        print("############################ Categorical Features Transformation: START ")
        print("\n")
        
        
        if 'ordinal_encoding' in categorical:
            print("ordinal_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = ordinal_encoding(features_ordinal, X, y, X_train, y_train, X_test, y_test)
            
        if 'one_hot_encoding' in categorical:
            print("one_hot_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = one_hot_encoding(features_one_hot, X, y, X_train, y_train, X_test, y_test)

        if 'label_encoding' in categorical:
            print("label_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = label_encoding(features_label, X, y, X_train, y_train, X_test, y_test)
            
        if 'helmert_encoding' in categorical:
            print("helmert_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = helmert_encoding(features_helmert, X, y, X_train, y_train, X_test, y_test)
                    
        if 'binary_encoding' in categorical:
            print("binary_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = binary_encoding(features_binary, X, y, X_train, y_train, X_test, y_test)
            
        if 'frequency_encoding' in categorical:
            print("frequency_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = frequency_encoding(features_frequency, X, y, X_train, y_train, X_test, y_test)
            
        if 'mean_encoding' in categorical:
            print("mean_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = mean_encoding(features_mean, X, y, X_train, y_train, X_test, y_test)
           
        if 'sum_encoding' in categorical:
            print("sum_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = sum_encoding(features_sum, X, y, X_train, y_train, X_test, y_test)
                      
        if 'weightofevidence_encoding' in categorical:
            print("weightofevidence_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = weightofevidence_encoding(features_weight, X, y, X_train, y_train, X_test, y_test)
           
        if 'probability_ratio_encoding' in categorical:
            print("probability_ratio_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = probability_ratio_encoding(features_proba_ratio, X, y, X_train, y_train, X_test, y_test)
                    
        if 'hashing_encoding' in categorical:
            print("hashing_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = hashing_encoding(features_hashing, X, y, X_train, y_train, X_test, y_test)
           
        if 'backward_difference_encoding' in categorical:
            print("backward_difference_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = backward_difference_encoding(features_backward, X, y, X_train, y_train, X_test, y_test)
           
        if 'leave_one_out_encoding' in categorical:
            print("leave_one_out_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = leave_one_out_encoding(features_leave_one_out, X, y, X_train, y_train, X_test, y_test)
           
        if 'james_stein_encoding' in categorical:
            print("james_stein_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = james_stein_encoding(features_james_stein, X, y, X_train, y_train, X_test, y_test)
                       
        if 'm_estimator_encoding' in categorical:
            print("m_estimator_encoding method was selected")
            X_cat, X_num, X_train_cat, X_train_num, X_test_cat, X_test_num = m_estimator_encoding(features_m, X, y, X_train, y_train, X_test, y_test)
            
        #print("Printing categorical and numerical data\n")
        #print("Printing X with categorical data:\n")
        #print(X_cat)
        #print("X with numerical data:\n")
        #print(X_num)
        #print("Printing y:\n")
        #print(y)
        #print("X_train with categorical data:\n")
        #print(X_train_cat)
        #print("X_train with numerical data:\n")
        #print(X_train_num)
        #print("Printing y_train:\n")
        #print(y_train)
        #print("X_test with categorical data:\n")
        #print(X_test_cat)
        #print("X_test with numerical data:\n")
        #print(X_test_num)
        #print("Printing y_test:\n")
        #print(y_test)
        
        #print("Concatenation of the transformed data (categorical + numerical)")
        X = pd.concat([X_cat, X_num], axis = 1)
        X_train = pd.concat([X_train_cat, X_train_num], axis = 1)
        X_test = pd.concat([X_test_cat, X_test_num], axis = 1)
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        
        print("\n")
        print("Printing X and y following categorical transformation: ")
        print("\n")
        print(X)
        print("\n")
        print(y)
    
        print("\n")
        print("Printing X_train and y_train following categorical transformation:\n ")
        print("X_train: \n")
        print(X_train)
        print("y_train: \n")
        print(y_train)
    
        print("\n")
        print("Printing X_test and y_test following categorical transformation:\n ")
        print("X_test: \n")
        print(X_test)
        print("y_test: \n")
        print(y_test)
        print("\n")
        
        print("############################ Categorical Features Transformation: END ")
        print("\n")
    
    else:
        print("No Categorical Features Transformation Selected \n")
                        
    if(rescaling is not None):
        # See feature_scaling.py to read explanations regarding the functions, inputs and outputs.
        # In the following lines, X_train and X_test are rescaled separately.
        print("############################ Features Scaling: START ")
        
        if rescaling == 'standard_scaler':
            print("\n")
            print("StandardScaler selected")
            print("\n")
            X_train=standardscaler(X_train)
            X_test=standardscaler(X_test)
            print("OOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKO")
            scaling_method = "StandardScaler"
            
        if rescaling == 'minmax_scaler':
            print("\n")
            print("MinMaxScaler selected")
            print("\n")
            X_train=minmaxscaler(X_train)
            X_test=minmaxscaler(X_test)
            scaling_method = "MinMaxScaler"
                
        if rescaling == 'maxabs_scaler':
            print("\n")
            print("MaxAbsScaler selected")
            print("\n")
            X_train=maxabsscaler(X_train)
            X_test=maxabsscaler(X_test)
            scaling_method = "MaxAbsScaler"
                
        if rescaling == 'robust_scaler':
            print("\n")
            print("RobustScaler selected")
            print("\n")
            X_train=robustscaler(X_train)
            X_test=robustscaler(X_test)
            scaling_method = "RobustScaler"
                
        if rescaling == 'normalizer':
            print("\n")
            print("Normalizer selected")
            print("\n")
            X_train=normalize(X_train)
            X_test=normalize(X_test)
            scaling_method = "Normalizer"
                
        if rescaling == 'log_transformation':
            print("\n")
            print("Log transformation selected\n")
            print("\n")
            X_train=logtransformation(X_train)
            X_test=logtransformation(X_test)
            scaling_method = "Log"
                
        if rescaling == 'square_root_transformation':
            print("\n")
            print("Square root transformation selected")
            print("\n")
            X_train=squareroottransformation(X_train)
            X_test=squareroottransformation(X_test)
            scaling_method = "SquareRoot"
                
        if rescaling == 'reciprocal_transformation':
            print("\n")
            print("Reciprocal transformation selected")
            print("\n")
            X_train=reciprocaltransformation(X_train)
            X_test=reciprocaltransformation(X_test)
            scaling_method = "Reciprocal"

        if rescaling == 'box_cox':
            print("\n")
            print("Box-Cox transformation selected")
            print("\n")
            X_train=boxcoxtransformation(X_train)
            X_test=boxcoxtransformation(X_test)
            scaling_method = "Box-Cox"
                 
        if rescaling == 'yeo_johnson':
            print("\n")
            print("Yeo-Johnson transformation selected")
            print("\n")
            X_train=yeojohnsontransformation(X_train)
            X_test=yeojohnsontransformation(X_test)
            scaling_method = "Yeo-Johnson"
                
        if rescaling == 'quantile_gaussian':
            print("\n")
            print("Quantile Transformation Gaussian selected")
            print("\n")
            X_train=quantiletransformationgaussian(X_train)
            X_test=quantiletransformationgaussian(X_test)
            scaling_method = "Quantile-Gaussian"
            
        if rescaling == 'quantile_uniform':
            print("\n")
            print("Quantile Transformation Uniform selected")
            print("\n")
            X_train=quantiletransformationuniform(X_train)
            X_test=quantiletransformationuniform(X_test)
            scaling_method = "Quantile-Uniform"
            
        print("\n")
        print("Printing X_train after rescaling:")
        print("\n")
        print(X_train)
    
        print("\n")
        print("Printing X_test after rescaling")
        print("\n")
        print(X_test)
        
        print("\n")
        print("############################ Features Scaling: END")
        print("\n")
        
    else:
        print("No Rescaling method selected")
        scaling_method = 'None'
        
    if(features_extraction is not None):
    # See feature_extraction.py to read explanations regarding the functions, inputs and outputs.
    # In the following lines, X_train and X_test reduced.
        print("\n")
        print("############################ Features Extraction: START")
        print("\n")
        if features_extraction == 'pca':
            extraction_method = 'PCA'
            print(extraction_method)
    
            print("PCA Reduction of X_train\n")
            X_train = pca(X_train, number_components)
            print(X_train)
            print("\n")
            
            print("PCA Reduction of X_test\n")
            X_test = pca(X_test, number_components)
            print(X_test)
    
        if features_extraction == 'icawithpca':
            extraction_method = 'ICA with PCA'
            
            print("ICA with PCA Reduction of X_train\n")
            X_train = icawithpca(X_train, number_components)
            print(X_train)
            print("\n")
            
            print("ICA with PCA Reduction of X_test\n")
            X_test = icawithpca(X_test, number_components)
            print(X_test)
    
        if features_extraction == 'ica':
            extraction_method = 'ICA without PCA'

            print("ICA without PCA Reduction of X_train\n")
            X_train = ica(X_train, number_components)
            print(X_train)
            print("\n")

            print("ICA without PCA Reduction of X_test\n")
            X_test = ica(X_test, number_components)
            print(X_test)
            
        if features_extraction == 'lda_extraction':
            extraction_method = 'LDA'
            X_train, X_test = lda_extraction(X_train, y_train, X_test, y_test, number_components)
            print("LDA Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("LDA Reduction of X_test\n")
            print(X_test)
        
        if features_extraction == 'random_projection':
            extraction_method = 'Random Projection'
            X_train, X_test = random_projection(X_train, y_train, X_test, y_test, number_components)
            print("Random Projection Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Random Projection Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'truncatedSVD':
            extraction_method = 'TruncatedSVD'
            X_train, X_test = truncatedSVD(X_train,y_train,X_test,y_test, number_components)
            print("Truncated SVD Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Truncated SVD Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'isomap':
            extraction_method = 'Isomap'
            X_train, X_test = isomap(X_train,y_train,X_test,y_test, number_components, n_neighbors)
            print("Isomap Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Isomap Reduction of X_test\n")
            print(X_test)

        if features_extraction == 'standard_lle':
            extraction_method = 'Standard LLE'
            X_train, X_test = standard_lle(X_train,y_train,X_test,y_test, number_components, n_neighbors)
            print("Standard LLE Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Standard LLE Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'modified_lle':
            extraction_method = 'Modified LLE'
            X_train, X_test = modified_lle(X_train,y_train,X_test,y_test, number_components, n_neighbors)
            print("Modified LLE Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Modified LLE Reduction of X_test\n")
            print(X_test)

        if features_extraction == 'hessian_lle':
            extraction_method = 'Hessian LLE'
            X_train, X_test = hessian_lle(X_train,y_train,X_test,y_test, number_components, n_neighbors)
            print("Hessian LLE Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Hessian LLE Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'ltsa_lle':
            extraction_method = 'LTSA LLE'
            X_train, X_test = ltsa_lle(X_train,y_train,X_test,y_test, number_components, n_neighbors)
            print("LTSA LLE Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("LTSA LLE Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'mds':
            extraction_method = 'MDS'
            X_train, X_test = mds(X_train,y_train,X_test,y_test, number_components)
            print("MDS Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("MDS Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'spectral':
            extraction_method = 'Spectral'
            X_train, X_test = spectral(X_train,y_train,X_test,y_test, number_components)
            print("Spectral Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("Spectral Reduction of X_test\n")
            print(X_test)
            
        if features_extraction == 'tsne':
            extraction_method = 't-SNE'
            X_train, X_test = tsne(X_train,y_train,X_test,y_test, number_components)
            print("t-SNE Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("t-SNE Reduction of X_test\n")
            print(X_test)

        if features_extraction == 'nca':
            extraction_method = 'NCA'
            X_train, X_test = nca(X_train,y_train,X_test,y_test, number_components)
            print("NCA Reduction of X_train\n")
            print(X_train)
            print("\n")
            print("NCA Reduction of X_test\n")
            print(X_test)
        print("\n")
        print("############################ Features Extraction: END ")
        print("\n")
        
    else:
        print("No extraction method selected")
        extraction_method = 'None'
        print("\n")

    print("############################ Features Selection: START")
    print("\n")
        
    if(feature_selection is not None):

    # Filter methods
    
        # Variance Threshold method
        if feature_selection == 'variance_threshold':
            print("variance_threshold feature selection method was selected")
            if 'all' in features_to_variance :
                X_train = variance_threshold(X_train, var_threshold)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
            else:
                X_train = variance_threshold(X_train[features_to_variance], var_threshold)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

        # Perform a chi-square test to the samples to retrieve only the k best features
        if feature_selection == 'chi_square':
                print("chi_square feature selection method was selected")
                X_train = chi_square(X_train, y_train, k_features)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Create an SelectKBest object to select features with k best ANOVA F-Values (for classification)
        if feature_selection == 'anova_f_c':
                print("anova_f_c feature selection method was selected")
                X_train = anova_f_c(X_train, y_train, k_features)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Create an SelectKBest object to select features with k best ANOVA F-Values (for regression)
        if feature_selection == 'anova_f_r':
                print("anova_f_r feature selection method was selected")
                X_train = anova_f_r(X_train, y_train, k_features)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
        
        # Pearson correlation coefficient: Keep variables with correlation greater than coefficient of correlation
        # X_test will just drop features from the new X_train
        if feature_selection == 'pearson':
                print("pearson feature selection method was selected")
                X_train = pearson(X_train, y_train, cc_features, cc_target)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
      
    # Wrapper methods
    
        # Forward Stepwise
        if feature_selection == 'forward_stepwise':
                print("forward_stepwise feature selection method was selected")
                X_train = forward_stepwise(X_train, y_train, wrapper_classifier, k_features, cv)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

        # Backward Elimination
        if feature_selection == 'backward_elimination':
                print("backward_elimination feature selection method was selected")
                X_train = backward_elimination(X_train, y_train, wrapper_classifier, k_features, cv)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Exhaustive Feature Selection
        if feature_selection == 'exhaustive':
                print("exhaustive feature selection method was selected")
                X_train = exhaustive(X_train, y_train, wrapper_classifier, min_features, max_features)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
    # Embedded methods
    
        # Lasso
        if feature_selection == 'lasso':
                print("lasso feature selection method was selected")
                X_train = lasso(X_train, y_train, lasso_alpha)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

        # Selecting features with regularization embedded into machine learning algorithms
        if feature_selection == 'feat_reg_ml':
                print("feat_reg_ml feature selection method was selected")
                X_train = feat_reg_ml(X_train, y_train, ml_penalty)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

        # Linear Regression Feature Importance
        if feature_selection == 'embedded_linear_regression':
                print("embedded_linear_regression feature selection method was selected")
                X_train = embedded_linear_regression(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Logistic Regression Feature Importance
        if feature_selection == 'embedded_logistic_regression':
                print("embedded_logistic_regression feature selection method was selected")
                X_train = embedded_logistic_regression(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Decision Tree Regressor Feature Importance
        if feature_selection == 'embedded_decision_tree_regressor':
                print("embedded_decision_tree_regressor feature selection method was selected")
                X_train = embedded_decision_tree_regressor(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Decision Tree Classifier Feature Importance
        if feature_selection == 'embedded_decision_tree_classifier':
                print("embedded_decision_tree_classifier feature selection method was selected")
                X_train = embedded_decision_tree_classifier(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
        
        # Random Forest Regressor Feature Importance
        if feature_selection == 'embedded_random_forest_regressor':
                print("embedded_random_forest_regressor feature selection method was selected")
                X_train = embedded_random_forest_regressor(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Random Forest Classifier Feature Importance
        if feature_selection == 'embedded_random_forest_classifier':
                print("embedded_random_forest_classifier feature selection method was selected")
                X_train = embedded_random_forest_classifier(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

        # Permutation Feature Importance for Regression
        if feature_selection == 'embedded_permutation_regression':
                print("embedded_permutation_regression feature selection method was selected")
                X_train = embedded_permutation_regression(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
                
        # Permutation Feature Importance for Classification
        if feature_selection == 'embedded_permutation_classification':
                print("embedded_permutation_classification feature selection method was selected")
                X_train = embedded_permutation_classification(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

        # Permutation Features Importance for Regression
        if feature_selection == 'embedded_xgboost_regression':
                print("embedded_xgboost_regression feature selection method was selected")
                X_train = embedded_xgboost_regression(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)
        
        # Permutation Features Importance for Classification
        if feature_selection == 'embedded_xgboost_classification':
                print("embedded_xgboost_classification feature selection method was selected")
                X_train = embedded_xgboost_classification(X_train, y_train, k_features, output_folder)
                X_test = pd.DataFrame(data = X_test, columns = X_train.columns)

    
    else:
        print("No Feature Selection Method Selected\n")

    print("\n")
    print("############################ Features Selection: END ")
    print("\n")
        
    """
    
    A L G O R I T H M S  F O R  C L A S S I F I C A T I O N
    
    """
    
    if classification_algorithms is not None:
    
        if cv is None:
            cv = 0
            
        print("\n")
        #print("Creating an empty DataFrame for merging at the end of the procedure the classification results\n")
        classification_results = pd.DataFrame()  # Creating an empty dataframe for merging at the end the classification results
        gpu_classification_results = pd.DataFrame() # Creating an empty dataframe for merging at the end the classification results done with GPUs
        print("\n")
        
        """
        Multi-layer Perceptron Neural Networks
        """
        try:
        # MLP
            if 'mlp_neural_network' in classification_algorithms:
                new_row_scaling = {'MLP_neural_network':["%s"%scaling_method]}
                new_row_missing = {'MLP_neural_network':["%s"%missing_method]}
                new_row_extraction = {'MLP_neural_network':["%s"%extraction_method]}
                df_technique = pd.DataFrame(new_row_scaling, columns=['MLP_neural_network'], index=["Rescaling Method"])
                df_missing = pd.DataFrame(new_row_missing, columns=['MLP_neural_network'], index=["Missing Method"])
                df_extraction = pd.DataFrame(new_row_extraction, columns=['MLP_neural_network'], index=["Extraction Method"])
                df_results = mlp_neural_network(X, X_train, X_test, y, y_train, y_test, cv, max_iter, hidden_layer_sizes, mlp_activation, solver, alpha, mlp_learning_rate, learning_rate_init, output_folder)
                mlp_neural_network_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
                classification_results = pd.concat([classification_results,mlp_neural_network_output], axis=1, join='outer')
                print("MLP Neural Network: OK\n")
        except Exception:
            pass
            
        """
        Multi-layer Perceptron Neural Networks: Auto
        """
        try:
            # MLP Auto
            if 'mlp_neural_network_auto' in classification_algorithms:
                new_row_scaling = {'mlp_neural_network_auto':["%s"%scaling_method]}
                new_row_missing = {'mlp_neural_network_auto':["%s"%missing_method]}
                new_row_extraction = {'mlp_neural_network_auto':["%s"%extraction_method]}
                df_technique = pd.DataFrame(new_row_scaling, columns=['mlp_neural_network_auto'], index=["Rescaling Method"])
                df_missing = pd.DataFrame(new_row_missing, columns=['mlp_neural_network_auto'], index=["Missing Method"])
                df_extraction = pd.DataFrame(new_row_extraction, columns=['mlp_neural_network_auto'], index=["Extraction Method"])
                df_results = mlp_neural_network_auto(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
                mlp_neural_network_auto_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
                classification_results = pd.concat([classification_results,mlp_neural_network_auto_output], axis=1, join='outer')
                print("MLP Neural Network: OK\n")
        except Exception:
            pass
        
        """
        Support Vector Machine for classification
        """
       
        # SVM with linear kernel
        if 'svm_linear' in classification_algorithms:
            print("SVM with linear kernel: OK\n")
            new_row_scaling = {'SVM_linear':["%s"%scaling_method]}
            new_row_missing = {'SVM_linear':["%s"%missing_method]}
            new_row_extraction = {'SVM_linear':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['SVM_linear'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['SVM_linear'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['SVM_linear'], index=["Extraction Method"])
            df_results = svm_linear(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            svm_linear_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,svm_linear_output], axis=1, join='outer')
            

        # SVM with rbf kernel
        if 'svm_rbf' in classification_algorithms:
            print("SVM with rbf kernel: OK\n")
            new_row_scaling = {'SVM_rbf':["%s"%scaling_method]}
            new_row_missing = {'SVM_rbf':["%s"%missing_method]}
            new_row_extraction = {'SVM_rbf':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['SVM_rbf'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['SVM_rbf'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['SVM_rbf'], index=["Extraction Method"])
            df_results = svm_rbf(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            svm_rbf_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,svm_rbf_output], axis=1, join='outer')


        # SVM with sigmoid kernel
        if 'svm_sigmoid' in classification_algorithms:
            print("SVM with sigmoid kernel: OK\n")
            new_row_scaling = {'SVM_sigmoid':["%s"%scaling_method]}
            new_row_missing = {'SVM_sigmoid':["%s"%missing_method]}
            new_row_extraction = {'SVM_sigmoid':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['SVM_sigmoid'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['SVM_sigmoid'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['SVM_sigmoid'], index=["Extraction Method"])
            df_results = svm_sigmoid(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            svm_sigmoid_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,svm_sigmoid_output], axis=1, join='outer')

            
        # SVM with polynomial kernel
        if 'svm_poly' in classification_algorithms:
            print("SVM with polynomial kernel: OK\n")
            new_row_scaling = {'SVM_poly':["%s"%scaling_method]}
            new_row_missing = {'SVM_poly':["%s"%missing_method]}
            new_row_extraction = {'SVM_poly':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['SVM_poly'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['SVM_poly'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['SVM_poly'], index=["Extraction Method"])
            df_results = svm_poly(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            svm_poly_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,svm_poly_output], axis=1, join='outer')

            
        """
        Linear and Quadratic Discriminant Analysis
        """
        
        # Linear discriminant analysis
        if 'lda' in classification_algorithms:
            print("Linear discriminant analysis: OK\n")
            new_row_scaling = {'lda':["%s"%scaling_method]}
            new_row_missing = {'lda':["%s"%missing_method]}
            new_row_extraction = {'lda':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['lda'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['lda'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['lda'], index=["Extraction Method"])
            df_results = lda(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            lda_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,lda_output], axis=1, join='outer')


        # Quadratic discriminant analysis
        if 'qda' in classification_algorithms:
            print("Quadratic discriminant analysis: OK\n")
            new_row_scaling = {'qda':["%s"%scaling_method]}
            new_row_missing = {'qda':["%s"%missing_method]}
            new_row_extraction = {'qda':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['qda'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['qda'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['qda'], index=["Extraction Method"])
            df_results = qda(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            qda_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,qda_output], axis=1, join='outer')


        """
        Naive Bayes
        """
        
        # Gaussian Naive Bayes
        if 'gnb' in classification_algorithms:
            print("Gaussian Naive Bayes: OK \n")
            new_row_scaling = {'gnb':["%s"%scaling_method]}
            new_row_missing = {'gnb':["%s"%missing_method]}
            new_row_extraction = {'gnb':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gnb'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gnb'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gnb'], index=["Extraction Method"])
            df_results = gnb(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            gnb_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,gnb_output], axis=1, join='outer')


        # Multinomial Naive Bayes
        if 'mnb' in classification_algorithms:
            print("Multinomial Naive Bayes: OK \n")
            new_row_scaling = {'mnb':["%s"%scaling_method]}
            new_row_missing = {'mnb':["%s"%missing_method]}
            new_row_extraction = {'mnb':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['mnb'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['mnb'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['mnb'], index=["Extraction Method"])
            df_results = mnb(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            mnb_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,mnb_output], axis=1, join='outer')


        """
        K-Neighbors
        """

        # K-Neighbors
        if 'kneighbors' in classification_algorithms:
            print("K-Neighbors: OK\n")
            new_row_scaling = {'kneighbors':["%s"%scaling_method]}
            new_row_missing = {'kneighbors':["%s"%missing_method]}
            new_row_extraction = {'kneighbors':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['kneighbors'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['kneighbors'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['kneighbors'], index=["Extraction Method"])
            df_results = kneighbors(X, X_train, X_test, y, y_train, y_test, cv, n_neighbors, output_folder)
            kneighbors_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,kneighbors_output], axis=1, join='outer')


        """
        Stochastic Gradient Descent
        """
        
        # Stochastic Gradient Descent
        if 'sgd' in classification_algorithms:
            print("Stochastic Gradient Descent: OK \n")
            new_row_scaling = {'sgd':["%s"%scaling_method]}
            new_row_missing = {'sgd':["%s"%missing_method]}
            new_row_extraction = {'sgd':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['sgd'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['sgd'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['sgd'], index=["Extraction Method"])
            df_results = sgd(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            sgd_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,sgd_output], axis=1, join='outer')

            

        """
        Nearest Centroid Classifier
        """

        # Nearest Centroid Classifier
        if 'nearest_centroid' in classification_algorithms:
            print("Nearest Centroid Classifier: OK \n")
            new_row_scaling = {'nearest_centroid':["%s"%scaling_method]}
            new_row_missing = {'nearest_centroid':["%s"%missing_method]}
            new_row_extraction = {'nearest_centroid':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['nearest_centroid'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['nearest_centroid'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['nearest_centroid'], index=["Extraction Method"])
            df_results = nearest_centroid(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            nearest_centroid_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,nearest_centroid_output], axis=1, join='outer')

            
        """
        Trees
        """

        # Decision Tree
        if 'decision_tree' in classification_algorithms:
            print("Decision Tree: OK \n")
            new_row_scaling = {'decision_tree':["%s"%scaling_method]}
            new_row_missing = {'decision_tree':["%s"%missing_method]}
            new_row_extraction = {'decision_tree':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['decision_tree'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['decision_tree'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['decision_tree'], index=["Extraction Method"])
            df_results = decision_tree(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            decision_tree_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,decision_tree_output], axis=1, join='outer')

        
        # Random Forest
        if 'random_forest' in classification_algorithms:
            print("Random Forest: OK \n")
            new_row_scaling = {'random_forest':["%s"%scaling_method]}
            new_row_missing = {'random_forest':["%s"%missing_method]}
            new_row_extraction = {'random_forest':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['random_forest'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['random_forest'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['random_forest'], index=["Extraction Method"])
            df_results = random_forest(X, X_train, X_test, y, y_train, y_test, cv, n_estimators_forest, output_folder)
            random_forest_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,random_forest_output], axis=1, join='outer')


        # Extremely Randomized Trees
        if 'extra_trees' in classification_algorithms:
            print("Extremely Randomized Trees: OK \n")
            new_row_scaling = {'extra_trees':["%s"%scaling_method]}
            new_row_missing = {'extra_trees':["%s"%missing_method]}
            new_row_extraction = {'extra_trees':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['extra_trees'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['extra_trees'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['extra_trees'], index=["Extraction Method"])
            df_results = extra_trees(X, X_train, X_test, y, y_train, y_test, cv, n_estimators_extra, output_folder)
            extra_trees_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,extra_trees_output], axis=1, join='outer')


        # Multinomial Logistic Regression
        if 'logistic_regression' in classification_algorithms:
            print("Multinomial Logistic Regression: OK \n")
            new_row_scaling = {'logistic_regression':["%s"%scaling_method]}
            new_row_missing = {'logistic_regression':["%s"%missing_method]}
            new_row_extraction = {'logistic_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['logistic_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['logistic_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['logistic_regression'], index=["Extraction Method"])
            df_results = logistic_regression(X, X_train, X_test, y, y_train, y_test, cv, output_folder)
            logistic_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            classification_results = pd.concat([classification_results,logistic_regression_output], axis=1, join='outer')

        # Multinomial Logistic Regression with GPU
        if 'gpu_logistic_regression' in classification_algorithms:
            print("Multinomial Logistic Regression with GPU: OK \n")
            new_row_scaling = {'gpu_logistic_regression':["%s"%scaling_method]}
            new_row_missing = {'gpu_logistic_regression':["%s"%missing_method]}
            new_row_extraction = {'gpu_logistic_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gpu_logistic_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gpu_logistic_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gpu_logistic_regression'], index=["Extraction Method"])
            df_results = gpu_logistic_regression(X, X_train, X_test, y, y_train, y_test, cv, df.groupby('Target').count().shape[0], gpu_logistic_optimizer, gpu_logistic_epochs, gpu_logistic_loss)
            gpu_logistic_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            gpu_classification_results = pd.concat([gpu_classification_results,gpu_logistic_regression_output], axis=1, join='outer')


        # Multi-layer with GPU
        if 'gpu_mlp' in classification_algorithms:
            print("Multi-layer perceptron neural network with GPU: OK \n")
            new_row_scaling = {'gpu_mlp':["%s"%scaling_method]}
            new_row_missing = {'gpu_mlp':["%s"%missing_method]}
            new_row_extraction = {'gpu_mlp':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gpu_mlp'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gpu_mlp'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gpu_mlp'], index=["Extraction Method"])
            df_results = gpu_mlp(X, X_train, X_test, y, y_train, y_test, cv, df.groupby('Target').count().shape[0], gpu_mlp_activation, gpu_mlp_optimizer, gpu_mlp_epochs, gpu_mlp_loss)
            gpu_mlp_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            gpu_classification_results = pd.concat([gpu_classification_results,gpu_mlp_output], axis=1, join='outer')

        # RNN
        if 'gpu_rnn' in classification_algorithms:
            print("RNN with GPU: OK \n")
            new_row_scaling = {'gpu_rnn':["%s"%scaling_method]}
            new_row_missing = {'gpu_rnn':["%s"%missing_method]}
            new_row_extraction = {'gpu_rnn':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gpu_rnn'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gpu_rnn'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gpu_rnn'], index=["Extraction Method"])
            df_results = gpu_rnn(X, X_train, X_test, y, y_train, y_test, cv, rnn_units, rnn_activation, rnn_optimizer, rnn_loss, rnn_epochs)
            gpu_rnn_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            gpu_classification_results = pd.concat([gpu_classification_results,gpu_rnn_output], axis=1, join='outer')
            

        # Save the data
        if output_folder is not None:
            print(classification_results)
            classification_results.to_csv(output_folder+'classification_metrics.csv')
        else:
            print(classification_results)
        
        print("\n")
            
        if output_folder is not None:
            print(gpu_classification_results)
            gpu_classification_results.to_csv(output_folder+'gpu_classification_metrics.csv')
        else:
            print(gpu_classification_results)
        
        print("\n")
    
    
    """
    A L G O R I T H M S
    F O R
    R E G R E S S I O N
    """
    
    if regression_algorithms is not None:
    
        #print("Create an empty DataFrame for merging at the end the regression results")
        regression_results = pd.DataFrame()  # Creating an empty dataframe for merging at the end the regression results
        linear_regression_results = pd.DataFrame() # Creating an empty dataframe for merging at the end the linear regression results
    
        """
        Linear Regression
        """
        
        # Compute a linear regression model
        if 'linear_regression' in regression_algorithms:
            print("Linear Regression: OK \n")
            new_row_scaling = {'linear_regression':["%s"%scaling_method]}
            new_row_missing = {'linear_regression':["%s"%missing_method]}
            new_row_extraction = {'linear_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['linear_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['linear_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['linear_regression'], index=["Extraction Method"])
            df_results = linear_regression(X, X_train, X_test, y, y_train, y_test, output_folder)
            linear_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            linear_regression_results = pd.concat([linear_regression_results,linear_regression_output], axis=1, join='outer')

            
        # Compute with GPU a linear regression model
        if 'gpu_linear_regression' in regression_algorithms:
            print("Linear Regression with GPU: OK \n")
            new_row_scaling = {'gpu_linear_regression':["%s"%scaling_method]}
            new_row_missing = {'gpu_linear_regression':["%s"%missing_method]}
            new_row_extraction = {'gpu_linear_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gpu_linear_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gpu_linear_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gpu_linear_regression'], index=["Extraction Method"])
            df_results = gpu_linear_regression(X, X_train, X_test, y, y_train, y_test, activation, epochs, learning_rate, output_folder)
            gpu_linear_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            linear_regression_results = pd.concat([linear_regression_results,gpu_linear_regression_output], axis=1, join='outer')

        
        if output_folder is not None:
            print(linear_regression_results)
            linear_regression_results.to_csv(output_folder+'regression_metrics.csv')
        else:
            print(linear_regression_results)
            
        print("\n")
        
        """
        Support Vector Machine for regression
        """
        
        # SVR with linear kernel
        if 'svr_linear' in regression_algorithms:
            print("SVR with linear kernel: OK \n")
            new_row_scaling = {'svr_linear':["%s"%scaling_method]}
            new_row_missing = {'svr_linear':["%s"%missing_method]}
            new_row_extraction = {'svr_linear':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['svr_linear'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['svr_linear'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['svr_linear'], index=["Extraction Method"])
            df_results = svr_linear(X, X_train, X_test, y, y_train, y_test, output_folder)
            svr_linear_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,svr_linear_output], axis=1, join='outer')


        # SVR with rbf kernel
        if 'svr_rbf' in regression_algorithms:
            print("SVR with rbf kernel: OK \n")
            new_row_scaling = {'svr_rbf':["%s"%scaling_method]}
            new_row_missing = {'svr_rbf':["%s"%missing_method]}
            new_row_extraction = {'svr_rbf':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['svr_rbf'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['svr_rbf'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['svr_rbf'], index=["Extraction Method"])
            df_results = svr_rbf(X, X_train, X_test, y, y_train, y_test, output_folder)
            svr_rbf_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,svr_rbf_output], axis=1, join='outer')


        # SVR with sigmoid kernel
        if 'svr_sigmoid' in regression_algorithms:
            print("SVR with sigmoid kernel: OK \n")
            new_row_scaling = {'svr_sigmoid':["%s"%scaling_method]}
            new_row_missing = {'svr_sigmoid':["%s"%missing_method]}
            new_row_extraction = {'svr_sigmoid':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['svr_sigmoid'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['svr_sigmoid'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['svr_sigmoid'], index=["Extraction Method"])
            df_results = svr_sigmoid(X, X_train, X_test, y, y_train, y_test, output_folder)
            svr_sigmoid_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,svr_sigmoid_output], axis=1, join='outer')


        # SVR with polynomial kernel
        if 'svr_poly' in regression_algorithms:
            print("SVR with polynomial kernel: OK \n")
            new_row_scaling = {'svr_poly':["%s"%scaling_method]}
            new_row_missing = {'svr_poly':["%s"%missing_method]}
            new_row_extraction = {'svr_poly':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['svr_poly'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['svr_poly'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['svr_poly'], index=["Extraction Method"])
            df_results = svr_poly(X, X_train, X_test, y, y_train, y_test, output_folder)
            svr_poly_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,svr_poly_output], axis=1, join='outer')

        """
        MLP Neural Network for regression
        """
        
        # MLP
        if 'mlp_regression' in regression_algorithms:
            print("MLP Neural Network for Regression: OK \n")
            new_row_scaling = {'mlp_regression':["%s"%scaling_method]}
            new_row_missing = {'mlp_regression':["%s"%missing_method]}
            new_row_extraction = {'mlp_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['mlp_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['mlp_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['mlp_regression'], index=["Extraction Method"])
            df_results = mlp_regression(X, X_train, X_test, y, y_train, y_test, max_iter_r, hidden_layer_sizes_r, mlp_activation_r, solver_r, alpha_r, mlp_learning_rate_r, learning_rate_init_r, output_folder)
            mlp_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,mlp_regression_output], axis=1, join='outer')

        # MLP Auto
        if 'mlp_auto_regression' in regression_algorithms:
            print("MLP Neural Network for Regression AUTO: OK \n")
            new_row_scaling = {'mlp_auto_regression':["%s"%scaling_method]}
            new_row_missing = {'mlp_auto_regression':["%s"%missing_method]}
            new_row_extraction = {'mlp_auto_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['mlp_auto_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['mlp_auto_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['mlp_auto_regression'], index=["Extraction Method"])
            df_results = mlp_auto_regression(X, X_train, X_test, y, y_train, y_test, output_folder)
            mlp_auto_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,mlp_auto_regression_output], axis=1, join='outer')

        # MLP
        if 'gpu_mlp_regression' in regression_algorithms:
            print("MLP Neural Network for Regression GPU: OK \n")
            new_row_scaling = {'gpu_mlp_regression':["%s"%scaling_method]}
            new_row_missing = {'gpu_mlp_regression':["%s"%missing_method]}
            new_row_extraction = {'gpu_mlp_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gpu_mlp_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gpu_mlp_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gpu_mlp_regression'], index=["Extraction Method"])
            df_results = gpu_mlp_regression(X, X_train, X_test, y, y_train, y_test, gpu_mlp_epochs_r, gpu_mlp_activation_r, output_folder)
            gpu_mlp_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,gpu_mlp_regression_output], axis=1, join='outer')

        # RNN for regression
        if 'gpu_rnn_regression' in regression_algorithms:
            print("RNN for Regression GPU: OK \n")
            new_row_scaling = {'gpu_rnn_regression':["%s"%scaling_method]}
            new_row_missing = {'gpu_rnn_regression':["%s"%missing_method]}
            new_row_extraction = {'gpu_rnn_regression':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['gpu_rnn_regression'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['gpu_rnn_regression'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['gpu_rnn_regression'], index=["Extraction Method"])
            df_results = gpu_rnn_regression(X, X_train, X_test, y, y_train, y_test, rnn_units, rnn_activation, rnn_optimizer, rnn_loss, rnn_epochs, output_folder)
            gpu_rnn_regression_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            regression_results = pd.concat([regression_results,gpu_rnn_regression_output], axis=1, join='outer')
           
        if output_folder is not None:
            print(regression_results)
            regression_results.to_csv(output_folder+'regression_metrics.csv')
        else:
            print(regression_results)
    
    print("\n")
    print("PIPELINE FINISHED")
    
    """
    
    C O N V O L U T I O N A L Neural Networks
    
    """
    
    # Convolutional Neural Network
    if convolutional is not None:
        if 'conv2d' in convolutional:
            print("Convolution Neural Network (conv2d) with GPU if available: OK \n")
            df_results = conv2d(X, X_train, X_test, y, y_train, y_test, conv_activation, conv_kernel_size, conv_optimizer, conv_loss, conv_epochs, output_folder)
            print(f"Test accuracy: {df_results:.5f}")
            
    """
    
    Q U A N T U M
    A L G O R I T H M S
    
    """
    
    
    if quantum_algorithms is not None:
    
        quantum_results = pd.DataFrame()  # Creating an empty dataframe for merging at the end the quantum results
    
        """
        q_kernel_zz
        """

        # q_kernel_zz
        if 'q_kernel_zz' in quantum_algorithms:
            print("\n")
            print("q_kernel_zz: OK \n")
            new_row_scaling = {'q_kernel_zz':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_zz':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_zz':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_zz'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_zz'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_zz'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_zz(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_zz_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_zz_output], axis=1, join='outer')
        """
        q_kernel_default
        """
    
        # q_kernel_default
        if 'q_kernel_default' in quantum_algorithms:
            print("\n")
            print("q_kernel_default: OK \n")
            new_row_scaling = {'q_kernel_default':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_default':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_default':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_default'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_default'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_default'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_default(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_default_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_default_output], axis=1, join='outer')

        """
        q_kernel_8
        """
        # q_kernel_default
        if 'q_kernel_8' in quantum_algorithms:
            print("\n")
            print("q_kernel_8: OK \n")
            new_row_scaling = {'q_kernel_8':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_8':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_8':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_8'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_8'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_8'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_8(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_8_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_8_output], axis=1, join='outer')
        
        """
        q_kernel_9
        """
        # q_kernel_default
        if 'q_kernel_9' in quantum_algorithms:
            print("\n")
            print("q_kernel_9: OK \n")
            new_row_scaling = {'q_kernel_9':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_9':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_9':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_9'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_9'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_9'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_9(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_9_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_9_output], axis=1, join='outer')
        
        """
        q_kernel_10
        """
    
        # q_kernel_10
        if 'q_kernel_10' in quantum_algorithms:
            print("\n")
            print("q_kernel_10: OK \n")
            new_row_scaling = {'q_kernel_10':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_10':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_10':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_10'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_10'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_10'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_10(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_10_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_10_output], axis=1, join='outer')
        
        """
        q_kernel_11
        """
        # q_kernel_default
        if 'q_kernel_11' in quantum_algorithms:
            print("\n")
            print("q_kernel_11: OK \n")
            new_row_scaling = {'q_kernel_11':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_11':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_11':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_11'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_11'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_11'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_11(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_11_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_11_output], axis=1, join='outer')
        
        """
        q_kernel_12
        """
        # q_kernel_default
        if 'q_kernel_12' in quantum_algorithms:
            print("\n")
            print("q_kernel_12: OK \n")
            new_row_scaling = {'q_kernel_12':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_12':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_12':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_12'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_12'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_12'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_12(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_12_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_12_output], axis=1, join='outer')

        """
        q_kernel_training
        """
        # q_kernel_default
        if 'q_kernel_training' in quantum_algorithms:
            print("\n")
            print("q_kernel_training: OK \n")
            new_row_scaling = {'q_kernel_training':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_training':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_training':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_training'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_training'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_training'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_training(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, multiclass, output_folder)
            q_kernel_training_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_training_output], axis=1, join='outer')
 
        """
        Q_Kernel_zz_pegasos
        """
        
        # Q_Kernel_zz_pegasos
        if 'q_kernel_zz_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_zz_pegasos: OK \n")
            new_row_scaling = {'q_kernel_zz_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_zz_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_zz_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_zz_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_zz_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_zz_pegasos'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = Q_Kernel_zz_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_zz_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_zz_pegasos_output], axis=1, join='outer')
        
        
        """
        q_kernel_default_pegasos
        """
        
        # q_kernel_default_pegasos
        if 'q_kernel_default_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_default_pegasos: OK \n")
            new_row_scaling = {'q_kernel_default_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_default_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_default_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_default_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_default_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_default_pegasos'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_default_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_default_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_default_pegasos_output], axis=1, join='outer')
            
        """
        q_kernel_8_pegasos
        """
        
        # q_kernel_8_pegasos
        if 'q_kernel_8_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_8_pegasos: OK \n")
            new_row_scaling = {'q_kernel_8_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_8_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_8_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_8_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_8_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_8_pegasos'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_8_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_8_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_8_pegasos_output], axis=1, join='outer')

        """
        q_kernel_9_pegasos
        """
        
        # q_kernel_9_pegasos
        if 'q_kernel_9_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_9_pegasos: OK \n")
            new_row_scaling = {'q_kernel_9_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_9_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_9_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_9_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_9_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_9_pegasos'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_9_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_9_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_9_pegasos_output], axis=1, join='outer')
            
        """
        q_kernel_10_pegasos
        """
        
        # q_kernel_10_pegasos
        if 'q_kernel_10_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_10_pegasos: OK \n")
            new_row_scaling = {'q_kernel_10_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_10_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_10_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_10_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_10_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_10_pegasos'], index=["Extraction Method"])
            
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_10_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_10_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_10_pegasos_output], axis=1, join='outer')
            
        """
        q_kernel_11_pegasos
        """
        
        # q_kernel_default
        if 'q_kernel_11_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_11_pegasos: OK \n")
            new_row_scaling = {'q_kernel_11_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_11_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_11_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_11_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_11_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_11_pegasos'], index=["Extraction Method"])
            
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_11_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_11_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_11_pegasos_output], axis=1, join='outer')
            
        """
        q_kernel_12_pegasos
        """
        
        # q_kernel_default
        if 'q_kernel_12_pegasos' in quantum_algorithms:
            print("\n")
            print("q_kernel_12_pegasos: OK \n")
            new_row_scaling = {'q_kernel_12_pegasos':["%s"%scaling_method]}
            new_row_missing = {'q_kernel_12_pegasos':["%s"%missing_method]}
            new_row_extraction = {'q_kernel_12_pegasos':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_kernel_12_pegasos'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_kernel_12_pegasos'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_kernel_12_pegasos'], index=["Extraction Method"])
            
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_kernel_12_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension, reps, ibm_account, quantum_backend, C, num_steps, output_folder)
            q_kernel_12_pegasos_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_kernel_12_pegasos_output], axis=1, join='outer')
            
        """
        q_twolayerqnn
        """
        
        # q_twolayerqnn
        if 'q_twolayerqnn' in quantum_algorithms:
            print("\n")
            print("q_twolayerqnn: OK \n")
            new_row_scaling = {'q_twolayerqnn':["%s"%scaling_method]}
            new_row_missing = {'q_twolayerqnn':["%s"%missing_method]}
            new_row_extraction = {'q_twolayerqnn':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_twolayerqnn'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_twolayerqnn'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_twolayerqnn'], index=["Extraction Method"])
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_twolayerqnn(X, X_train, X_test, y, y_train, y_test, feature_dimension, reps, ibm_account, quantum_backend, cv, output_folder)
            q_twolayerqnn_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_twolayerqnn_output], axis=1, join='outer')
            
        # q_circuitqnn
        if 'q_circuitqnn' in quantum_algorithms:
            print("\n")
            print("q_circuitqnn: OK \n")
            new_row_scaling = {'q_circuitqnn':["%s"%scaling_method]}
            new_row_missing = {'q_circuitqnn':["%s"%missing_method]}
            new_row_extraction = {'q_circuitqnn':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_circuitqnn'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_circuitqnn'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_circuitqnn'], index=["Extraction Method"])
            number_classes = len(df.groupby('Target').size().index) # Number of classes
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_circuitqnn(X, X_train, X_test, y, y_train, y_test, number_classes, feature_dimension, reps, ibm_account, quantum_backend, cv, output_folder)
            q_circuitqnn_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_circuitqnn_output], axis=1, join='outer')
                
        # q_vqc
        if 'q_vqc' in quantum_algorithms:
            print("\n")
            print("q_vqc: OK \n")
            new_row_scaling = {'q_vqc':["%s"%scaling_method]}
            new_row_missing = {'q_vqc':["%s"%missing_method]}
            new_row_extraction = {'q_vqc':["%s"%extraction_method]}
            df_technique = pd.DataFrame(new_row_scaling, columns=['q_vqc'], index=["Rescaling Method"])
            df_missing = pd.DataFrame(new_row_missing, columns=['q_vqc'], index=["Missing Method"])
            df_extraction = pd.DataFrame(new_row_extraction, columns=['q_vqc'], index=["Extraction Method"])
            number_classes = len(df.groupby('Target').size().index) # Number of classes
            feature_dimension = X_train.shape[1] # Number of features
            df_results = q_vqc(X, X_train, X_test, y, y_train, y_test, number_classes, feature_dimension, reps, ibm_account, quantum_backend, cv, output_folder)
            q_vqc_output=pd.concat([df_technique, df_missing, df_extraction, df_results])
            quantum_results = pd.concat([quantum_results, q_vqc_output], axis=1, join='outer')
        
    
        if output_folder is not None:
            print(quantum_results)
            quantum_results.to_csv(output_folder+'quantum_metrics.csv')
        else:
            print(quantum_results)
    
    print("\n")
    print("PIPELINE FINISHED")
    


    
                
#if __name__ == '__ml_pipeline_function__':
#    ml_pipeline_function(inputs.df, missing_method)
    
    




