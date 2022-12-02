#!/usr/bin/python3
# regression_cpu.py
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

Regression algorithms:
    - linear_regression,
    - svr_linear,
    - svr_rbf,
    - svr_sigmoid,
    - svr_poly

"""


import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report


# Compute a linear regression model
def linear_regression(X, X_train, X_test, y, y_train, y_test, output_folder = None):
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    """
    Linear Regression

    The linear regression model is trained with X_train and y_train. We also print predicted outputs with X_test as new input data

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        

    Output:
        A DataFrame with the following metrics: Slope, Intercept, Root Mean Squared Error, R2 Score
        We also print prediction with X_test as new input data
    """
    from sklearn.linear_model import LinearRegression
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'linear_regression.joblib')
    
    # Model evaluation
    r2 = model.score(X_train,y_train)
    mse = mean_squared_error(y_train, y_pred)
    
    #results = [model.coef_, model.intercept_, mse, r2]
    results = [mse, r2]

    #metrics_dataframe = pd.DataFrame(results, index=["Slope", "Intercept", "MSE", "R-squared"], columns={'linear_regression'})
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'linear_regression'})

    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    
    print("Linear Regression Metrics\n")
    print("Slope: ", model.coef_)
    print("Intercept: %f"%model.intercept_)
    print("MSE: %f"%mse)
    print("R-squared: %f"%r2)
    print('\n')
    
    return metrics_dataframe

"""
Support Vector Machines for regression
"""

# SVR with linear kernel
def svr_linear(X, X_train, X_test, y, y_train, y_test, output_folder= None):
    """
    SVR with linear kernel

    The SVR is trained with X_train and y_train. We also print predicted outputs with X_test as new input data

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        

    Output:
        A DataFrame with the following metrics: Slope, Intercept, Root Mean Squared Error, R2 Score
        We also print prediction with X_test as new input data
    """

    model=svm.SVR(kernel='linear')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    # Model evaluation
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svr_linear.joblib')

    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'svr_linear'})
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    return metrics_dataframe
    

# SVM with RBF kernel
def svr_rbf(X, X_train, X_test, y, y_train, y_test, output_folder= None):
    """
    SVR with rbf kernel

    The SVR is trained with X_train and y_train. We also print predicted outputs with X_test as new input data

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        

    Output:
        A DataFrame with the following metrics: Slope, Intercept, Root Mean Squared Error, R2 Score
        We also print prediction with X_test as new input data
    """
    from sklearn.svm import SVR
    model=SVR(kernel='rbf')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    
    # Model evaluation
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svr_rbf.joblib')
    
    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'svr_rbf'})
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    return metrics_dataframe
    
# SVM with sigmoid kernel
def svr_sigmoid(X, X_train, X_test, y, y_train, y_test, output_folder= None):
    """
    SVR with sigmoid kernel

    The SVR is trained with X_train and y_train. We also print predicted outputs with X_test as new input data

    Inputs:
        X,y non splitted dataset separated by features (X) and target (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and target (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and target (y_test)
        

    Output:
        A DataFrame with the following metrics: Slope, Intercept, Root Mean Squared Error, R2 Score
        We also print prediction with X_test as new input data
    """

    model=svm.SVR(kernel='sigmoid')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    
    # Model evaluation
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svr_sigmoid.joblib')
    
    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'svr_sigmoid'})
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    return metrics_dataframe
    

# SVM with polynomial kernel
def svr_poly(X, X_train, X_test, y, y_train, y_test, output_folder= None):
    """
    SVR with polynomial kernel

    The SVR is trained with X_train and y_train. We also print predicted outputs with X_test as new input data

    Inputs:
        X,y non splitted dataset separated by features (X) and target (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and target (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and target (y_test)
        

    Output:
        A DataFrame with the following metrics: Slope, Intercept, Root Mean Squared Error, R2 Score
        We also print prediction with X_test as new input data
    """

    model=svm.SVR(kernel='poly')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    
    # Model evaluation
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svr_poly.joblib')
    
    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'svr_poly'})
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    return metrics_dataframe

# Multi-layer perceptron neural network
def mlp_regression(X, X_train, X_test, y, y_train, y_test, max_iter_r = None, hidden_layer_sizes_r = None, mlp_activation_r = None, solver_r = None, alpha_r = None, mlp_learning_rate_r = None, learning_rate_init_r = None, output_folder= None):

    """
    Multi-layer Perceptron for regression

    Inputs:
        X,y non splitted dataset separated by features (X) and target (y).
        X_train, y_train selected dataset to train the model separated by features (X_train) and target (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        max_iter: Maximum number of iterations (default= 200)
        hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
        mlp_activation: Activation function for the hidden layer ('identity', 'logistic', 'relu', 'softmax', 'tanh'). default=’relu’
        solver: The solver for weight optimization (‘lbfgs’, ‘sgd’, ‘adam’). default=’adam’
        alpha: Strength of the L2 regularization term (default=0.0001)
        mlp_learning_rate: Learning rate schedule for weight updates (‘constant’, ‘invscaling’, ‘adaptive’). default='constant'
        learning_rate_init: The initial learning rate used (for sgd or adam). It controls the step-size in updating the weights.
        
    Output:
        A DataFrame with the following metrics: Slope, Intercept, Root Mean Squared Error, R2 Score
        We also print prediction with X_test as new input data
            
    """
    model = MLPRegressor(max_iter = max_iter_r, hidden_layer_sizes = hidden_layer_sizes_r, activation = mlp_activation_r, solver = solver_r, alpha = alpha_r, learning_rate = mlp_learning_rate_r, learning_rate_init = learning_rate_init_r)
    print(X_train, y_train)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_train)
    
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'mlp_regression.joblib')
    
    # Model evaluation
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'mlp_regression'})
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    return metrics_dataframe
    
    
# Neural Network (MLP)
def mlp_auto_regression(X, X_train, X_test, y, y_train, y_test, output_folder= None):
    """
    Multi-layer Perceptron : Run automatically different hyperparameters combinations and return the best result

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        
    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """
    
    # Instantiate the estimator
    mlp_gs = MLPRegressor()
    parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive','invscaling'],
    }
    # Hyperparameters search
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    # Fit the estimator to the data
    model.fit(X_train, y_train) # X is train samples and y is the corresponding labels
    # Use the model to predict the last several labels
    y_pred = model.predict(X_train)
    
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'mlp_auto_regression.joblib')

    # Model evaluation
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'mlp_auto_regression'})
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
    return metrics_dataframe
                    
