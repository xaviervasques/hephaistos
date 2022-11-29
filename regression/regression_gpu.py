#!/usr/bin/python3
# regression_gpu.py
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

Regression algorithms using GPUs:
    - gpu_linear_regression

"""
import inputs

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.metrics import classification_report


# Linear Regression with GPU (if available)
def gpu_linear_regression(X, X_train, X_test, y, y_train, y_test, gpu_linear_activation, gpu_linear_epochs, gpu_linear_learning_rate, gpu_linear_loss, output_folder):

    """
    Linear Regression using available GPUs.
    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model speparated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        gpu_linear_activation, gpu_linear_epochs, gpu_linear_learning_rate, gpu_linear_loss: You can set gpu_linear_regression to 'yes' or 'no'. You need to select the number of epochs with an integer and the optimizer such as stochastic gradient descent (SGD(learning_rate = 1e-2) or 'sgd'), adam ('adam') or RMSprop ('RMSprop'). loss: The loss functions such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic loss 'categorical_crossentropy').

    Output:
        A DataFrame with the following metrics:
            - Root mean squared error (MSE)
            - R2 score

    """

    
    # Print number of GPUs available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # For future use, which devices the operations and tensors are assigned to (GPU, CPU)
    #tf.debugging.set_log_device_placement(True)

    # Creating a Sequential Model with TF2
    # Sequential Layer allows stacking of one layer on top of the other , enabling the data to flow through them
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # We have two inputs for our model: 'Salnty' and 'O2ml_L' features
    model = Sequential()
    model.add(Dense(1, input_dim = X_train.shape[1], activation = gpu_linear_activation))
    model.summary()

    # Optimizer and Gradient Descent
    # We use mini-batch gradient descent optimizer and mean square loss
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.losses import mse
    model.compile(optimizer=SGD(learning_rate=gpu_linear_learning_rate),loss=gpu_linear_loss)
    train = model.fit(X_train,y_train,epochs=gpu_linear_epochs)

    import matplotlib.pyplot as plt
    # Performance Analysis: loss over time
    # We should see that the loss is reduced over time
    plt.plot(train.history['loss'],label='loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(inputs.output_folder+'gpu_linear_regression')

    # Model prediction
    y_pred = model.predict(X_train)
    
    if output_folder is not None:
        model.save(output_folder+'gpu_linear_regression')


    # Extracting the weights and biases is achieved quite easily
    model.layers[0].get_weights()

    # We can save the weights and biases in separate variables
    weights = model.layers[0].get_weights()[0]
    bias = model.layers[0].get_weights()[1]

    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)



    # Compute and print predicted output with X_test as new input data
    print("\n")
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')

    # Printing metrics
    print("Linear Regression Metrics\n")
    print('Root mean squared error: ', mse)
    print('R2 score: ', r2)
    print("Intercept:", bias)
    print("Weights:",weights)
    print('\n')

    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'gpu_linear_regression'})
    
    return metrics_dataframe
    
def gpu_mlp_regression(X, X_train, X_test, y, y_train, y_test, gpu_mlp_epochs_r, gpu_mlp_activation_r, output_folder):
        
    """
    Multi-Layer perceptron using GPU

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        gpu_mlp_epochs: The number of epochs can also be choosen
        gpu_mlp_activation_r: The activation function such as softmax, sigmoid, linear or tanh.

    Output:
        A DataFrame with the following metrics:
            - Root mean squared error (MSE)
            - R2 score
            
    """
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from sklearn.metrics import classification_report
    
    # Print number of GPUs available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # For future use, which devices the operations and tensors are assigned to (GPU, CPU)
    #tf.debugging.set_log_device_placement(True)
        
    # Define number of classes and number of features to include in our model
    number_of_features = X_train.shape[1]

    # Model creation
    keras_model = Sequential()
    keras_model.add(Dense(number_of_features, input_shape=(number_of_features,), kernel_initializer='normal', activation=gpu_mlp_activation_r))
    keras_model.add(Dense(1, kernel_initializer='normal'))
    keras_model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Model prediction
    y_pred = keras_model.predict(X_train)
    
    if output_folder is not None:
        keras_model.save(output_folder+'gpu_mlp_regression')

    # Extracting the weights and biases is achieved quite easily
    keras_model.layers[0].get_weights()

    # We can save the weights and biases in separate variables
    weights = keras_model.layers[0].get_weights()[0]
    bias = keras_model.layers[0].get_weights()[1]

    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    # Compute and print predicted output with X_test as new input data
    print("\n")
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', keras_model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')

    # Printing metrics
    print("MLP for Regression Metrics on GPU \n")
    print('Root mean squared error: ', mse)
    print('R2 score: ', r2)
    print("Intercept:", bias)
    print("Weights:",weights)
    print('\n')

    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'gpu_mlp_regression'})
    
    return metrics_dataframe
    

def gpu_rnn_regression(X, X_train, X_test, y, y_train, y_test, rnn_units, rnn_activation, rnn_optimizer, rnn_loss, rnn_epochs, output_folder):
        
    """
    Recurrent Neural Network using GPUs if available

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        rnn_units: Positive integer, dimensionality of the output space.
        rnn_activation:  Activation function to use (softmax, sigmoid, linear or tanh)
        rnn_optimizer: Optimizer (adam, sgd, RMSprop)
        rnn_loss: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic loss ('categorical_crossentropy').
        rnn_epochs: Number (Integer) of epochs

    Output:
        A DataFrame with the following metrics:
            - Root mean squared error (MSE)
            - R2 score
            
    """

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
    from tensorflow.keras.optimizers import SGD


    # Parameters and Hyperparameters of the model
    rnn_units = rnn_units # Positive integer, dimensionality of the output space.
    rnn_activation = rnn_activation # Activation function to use.
    rnn_features = X_train.shape[1] # Number of features
    rnn_optimizer= rnn_optimizer # Optimizer
    rnn_loss = rnn_loss # Loss function
    rnn_epochs = rnn_epochs # Number of epochs


    # The LSTM architecture
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=rnn_units, activation=rnn_activation, input_shape = (rnn_features, 1)))
    model_lstm.add(Dense(units=1))
    # Compiling the model
    model_lstm.compile(optimizer=rnn_optimizer, loss=rnn_loss)
    model_lstm.fit(X_train, y_train, epochs=rnn_epochs)

    
    # Model prediction
    y_pred = model_lstm.predict(X_train)
    
    if output_folder is not None:
        model_lstm.save(output_folder+'gpu_rnn_regression')

    # Extracting the weights and biases is achieved quite easily
    model_lstm.layers[0].get_weights()

    # We can save the weights and biases in separate variables
    weights = model_lstm.layers[0].get_weights()[0]
    bias = model_lstm.layers[0].get_weights()[1]

    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    # Compute and print predicted output with X_test as new input data
    print("\n")
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model_lstm.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')

    # Printing metrics
    print("RNN on GPU \n")
    print('Root mean squared error: ', mse)
    print('R2 score: ', r2)
    print("Intercept:", bias)
    print("Weights:",weights)
    print('\n')

    results = [mse, r2]
    metrics_dataframe = pd.DataFrame(results, index=["MSE", "R-squared"], columns={'gpu_rnn_regression'})
        
    return metrics_dataframe








