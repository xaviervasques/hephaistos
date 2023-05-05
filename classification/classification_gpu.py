#!/usr/bin/python3
# classification_gpu.py
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

Classification algorithms used with GPUs :
    - gpu_logistic_regression
    - gpu_mlp
    - gpu_rnn
    - conv2d

"""

import warnings
warnings.filterwarnings('always')

import numpy as np
import pandas as pd
import tensorflow as tf

def gpu_logistic_regression(X, X_train, X_test, y, y_train, y_test, cv, number_of_classes,
                            gpu_logistic_optimizer, gpu_logistic_epochs, gpu_logistic_loss,
                            output_folder=None):
    """
    Perform Logistic Regression with GPU (if available).

    Inputs:
        X, y: Non-splitted dataset separated by features (X) and labels (y).
        X_train, y_train: Dataset to train the model, separated by features (X_train) and labels (y_train).
        X_test, y_test: Dataset to test the model, separated by features (X_test) and labels (y_test).
        cv: Number of k-folds for cross-validation.
        optimizer: Model optimizers such as stochastic gradient descent ('sgd'), adam ('adam') or RMSprop ('RMSprop').
        epochs: Number of epochs to train the model.
        loss: Loss functions such as mean squared error ('mse'), binary logarithmic loss ('binary_crossentropy')
              or multi-class logarithmic loss ('categorical_crossentropy').

    Output:
        A DataFrame with the following metrics:
            accuracy_score: Ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: How many of the actual positive cases we were able to predict correctly.
            f1_score: Harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.optimizers import SGD
    from sklearn.metrics import classification_report

    # Print number of GPUs available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Define the Keras model
    keras_model = Sequential()
    keras_model.add(Flatten(input_dim=X_train.shape[1]))
    keras_model.add(Dense(number_of_classes, activation='linear'))
                
    keras_model.compile(optimizer=gpu_logistic_optimizer,
                        loss=gpu_logistic_loss,
                        metrics=[gpu_logistic_loss])
    
    # Train the model
    keras_model.fit(X_train, y_train, epochs=gpu_logistic_epochs)

    # Evaluate the model on the test set
    keras_model.evaluate(X_test, y_test)

    # Make predictions on the test set
    y_keras_pred = keras_model.predict(X_test)
    y_keras_test = np.argmax(y_keras_pred, axis=1)
    
    # Save the trained model
    if output_folder is not None:
        keras_model.save(output_folder + 'gpu_logistic_regression')
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
    # Print classification report
    print("Classification report for logistic regression using GPUs (if available)\n")
    print(classification_report(y_test, y_keras_test))
    print("\n")
                            
    # Print predicted output and real values
    print('Predicted output with X_test as new input data:\n')
    print('Predictions:\n', y_keras_test)
    print('\n')
    print('Real values:\n', y_test)
    print('\n')

    # Calculate metrics and return them as a DataFrame
    results = [accuracy_score(y_test, y_keras_test),
               precision_score(y_test, y_keras_test, average='micro'),
               recall_score(y_test, y_keras_test,average='micro'),
               f1_score(y_test, y_keras_test,average='micro')]
               
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"], columns={'gpu_logistic_regression'})
    
    return metrics_dataframe
    
def gpu_mlp(X, X_train, X_test, y, y_train, y_test, cv, number_of_classes,
            gpu_mlp_activation, gpu_mlp_optimizer, gpu_mlp_epochs, gpu_mlp_loss,
            output_folder=None):
    """
    Multi-Layer Perceptron using GPU (if available).

    Inputs:
        X, y: Non-splitted dataset separated by features (X) and labels (y) for cross-validation.
        X_train, y_train: Dataset to train the model, separated by features (X_train) and labels (y_train).
        X_test, y_test: Dataset to test the model, separated by features (X_test) and labels (y_test).
        cv: Number of k-folds for cross-validation.
        number_of_classes: The number of labels.
        gpu_activation: Activation functions such as softmax, sigmoid, linear, relu or tanh.
        gpu_optimizer: Model optimizers such as stochastic gradient descent ('sgd'), adam ('adam') or RMSprop ('RMSprop').
        gpu_loss: Loss functions such as mean squared error ('mse'), binary logarithmic loss ('binary_crossentropy')
                  or multi-class logarithmic loss ('categorical_crossentropy').
        gpu_epochs: Number of epochs to train the model.

    Output:
        A DataFrame with the following metrics:
            accuracy_score: Ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: How many of the actual positive cases we were able to predict correctly.
            f1_score: Harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.optimizers import SGD
    from sklearn.metrics import classification_report

    # Print number of GPUs available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Define the Keras model
    keras_model = Sequential()
    keras_model.add(Dense(number_of_classes, activation=gpu_mlp_activation))
    keras_model.add(Flatten(input_dim=X_train.shape[1]))
    keras_model.compile(optimizer=gpu_mlp_optimizer,
                        loss=gpu_mlp_loss,
                        metrics=[gpu_mlp_loss])

    # Train the model
    keras_model.fit(X_train, y_train, epochs=gpu_mlp_epochs)

    # Evaluate the model on the test set
    keras_model.evaluate(X_test, y_test)

    # Save the trained model
    if output_folder is not None:
        keras_model.save(output_folder + 'gpu_mlp')

    # Make predictions on the test set
    y_keras_pred = keras_model.predict(X_test)
    y_keras_test = np.argmax(y_keras_pred, axis=1)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Print classification report
    print("Classification report for multi-layer perceptron using GPUs (if available)\n")
    print(classification_report(y_test, y_keras_test))
    print("\n")

    # Print predicted output and real values
    print('Predicted output with X_test as new input data:\n')
    print('Predictions:\n', y_keras_test)
    print('\n')
    print('Real values:\n', y_test)
    print('\n')

    # Calculate metrics and return them as a DataFrame
    results = [accuracy_score(y_test, y_keras_test),
               precision_score(y_test, y_keras_test,average='micro'),
               recall_score(y_test, y_keras_test,average='micro'),
               f1_score(y_test, y_keras_test,average='micro')]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"], columns={'gpu_mlp'})
    
    return metrics_dataframe

def gpu_rnn(X, X_train, X_test, y, y_train, y_test, cv, rnn_units, rnn_activation,
            rnn_optimizer, rnn_loss, rnn_epochs, output_folder=None):
    """
    Recurrent Neural Network using GPUs (if available).

    Inputs:
        X, y: Non-splitted dataset separated by features (X) and labels (y) for cross-validation.
        X_train, y_train: Dataset to train the model, separated by features (X_train) and labels (y_train).
        X_test, y_test: Dataset to test the model, separated by features (X_test) and labels (y_test).
        cv: Number of k-folds for cross-validation.
        rnn_units: Positive integer, dimensionality of the output space.
        rnn_activation: Activation function to use (softmax, sigmoid, linear, relu, or tanh).
        rnn_optimizer: Optimizer (adam, sgd, RMSprop).
        rnn_loss: Loss function such as mean squared error ('mse'), binary logarithmic loss ('binary_crossentropy'),
                  or multi-class logarithmic loss ('categorical_crossentropy').
        rnn_epochs: Number (integer) of epochs to train the model.

    Output:
        A DataFrame with the following metrics:
            accuracy_score: Ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: How many of the actual positive cases we were able to predict correctly.
            f1_score: Harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import SGD
    from sklearn.metrics import classification_report

    # Define the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=rnn_units, activation=rnn_activation, input_shape=(X_train.shape[1], 1)))
    model_lstm.add(Dense(units=1))

    # Compile the model
    model_lstm.compile(optimizer=rnn_optimizer, loss=rnn_loss)

    # Train the model
    model_lstm.fit(X_train, y_train, epochs=rnn_epochs)

    # Evaluate the model on the test set
    model_lstm.evaluate(X_test, y_test)

    # Save the trained model
    if output_folder is not None:
        model_lstm.save(output_folder + 'gpu_rnn')

    # Make predictions on the test set
    y_keras_pred = model_lstm.predict(X_test)
    y_keras_test = np.argmax(y_keras_pred, axis=1)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Print classification report
    print("Classification report for RNN using GPUs (if available)\n")
    print(classification_report(y_test, y_keras_test))
    print("\n")

    # Print predicted output and real values
    print('Predicted output with X_test as new input data:\n')
    print('Predictions:\n', y_keras_test)
    print('\n')
    print('Real values:\n', y_test)
    print('\n')

    # Calculate metrics and return them as a DataFrame
    results = [accuracy_score(y_test, y_keras_test),
               precision_score(y_test, y_keras_test, average='micro'),
               recall_score(y_test, y_keras_test, average='micro'),
               f1_score(y_test, y_keras_test, average='micro')]

    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"], columns={'gpu_rnn'})
    
    return metrics_dataframe

def conv2d(X, X_train, X_test, y, y_train, y_test, conv_activation, conv_kernel_size,
           conv_optimizer, conv_loss, conv_epochs, output_folder=None):
    """
    2D Convolutional Neural Network using GPUs (if available).

    Inputs:
        X, y: Non-splitted dataset separated by features (X) and labels (y) for cross-validation.
        X_train, y_train: Dataset to train the model, separated by features (X_train) and labels (y_train).
        X_test, y_test: Dataset to test the model, separated by features (X_test) and labels (y_test).
        conv_kernel_size: Size of the filter matrix for the convolution (conv_kernel_size x conv_kernel_size).
        conv_activation: Activation function to use (softmax, sigmoid, linear, relu, or tanh).
        conv_optimizer: Optimizer (adam, sgd, RMSprop).
        conv_loss: Loss function such as mean squared error ('mse'), binary logarithmic loss ('binary_crossentropy'),
                   or multi-class logarithmic loss ('categorical_crossentropy').
        conv_epochs: Number (integer) of epochs to train the model.

    Output:
        Accuracy of the model using (X_test and y_test)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten

    # Create the model using Sequential() to build it layer by layer
    model = Sequential()

    # Add model layers
    model.add(Conv2D(int(X_train.shape[0] / 1000), kernel_size=conv_kernel_size, activation=conv_activation,
                     input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(Conv2D(int(X_train.shape[0] / 2000), kernel_size=conv_kernel_size, activation=conv_activation))

    # Add a Flatten layer between the Convolutional layers and the Dense layer to connect them
    model.add(Flatten())

    # Add the Dense output layer with softmax activation to produce probabilities
    model.add(Dense(int(X_test.shape[0] / 1000), activation='softmax'))

    # Compile the model with an optimizer, loss function, and performance metric
    model.compile(optimizer=conv_optimizer, loss=conv_loss, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=conv_epochs)

    # Predict the first 10 images in the test set
    predictions = model.predict(X_test[:10])

    # Save the trained model
    if output_folder is not None:
        model.save(output_folder + 'conv2d')

    # Print predicted output and real values
    print("\nPrediction of the first 10 images from the test dataset:\n")
    print(predictions)
    print("\n")
    print("Actual values of the first 10 images from the test dataset:\n")
    print(y_test[:10])

    # Evaluate model performance on test data and return the accuracy
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc

