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

"""

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import numpy as np
import pandas as pd
import tensorflow as tf

# Logistic Regression with GPU (if available)
def gpu_logistic_regression(X, X_train, X_test, y, y_train, y_test, cv, number_of_classes, gpu_logistic_optimizer, gpu_logistic_epochs, gpu_logistic_loss, output_folder=None):

        
    """
    Logistic Regression with GPU (if available)

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)
        optimizer: The model optimizers such as stochastic gradient descent ('sgd'), adam ('adam') or RMSprop ('RMSprop').
        epochs: The number of epochs can also be choosen
        loss: The loss functions such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic loss ('categorical_crossentropy').

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.optimizers import SGD
    from sklearn.metrics import classification_report
    
    # Print number of GPUs available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # For future use, which devices the operations and tensors are assigned to (GPU, CPU)
    #tf.debugging.set_log_device_placement(True)
    
    

    # Define number of classes and number of features to include in our model
    number_of_classes = number_of_classes
    number_of_features = X_train.shape[1]

    keras_model = Sequential()
    keras_model.add(Flatten(input_dim=number_of_features))
    keras_model.add(Dense(number_of_classes, activation='linear'))
                
    keras_model.compile(optimizer = gpu_logistic_optimizer,
                    loss = gpu_logistic_loss,
                    metrics = [gpu_logistic_loss])
    
    keras_model.fit(X_train, y_train, epochs=gpu_logistic_epochs)

    keras_model.evaluate(X_test, y_test) # loss, sparse_categorical_accuracy

    # Predicting X_test data with the created model
    y_keras_pred = keras_model.predict(X_test)
    y_keras_test = np.argmax(y_keras_pred,axis=1) #Make labels back
    
    if output_folder is not None:
        keras_model.save(output_folder+'gpu_logistic_regression')
    
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        
    print("Classfication report for logistic regression using GPUs (if available)\n")
    print(classification_report(y_test, y_keras_test))
    print("\n")
                            
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', y_keras_test)
    print('\n')
    print('Real values: \n', y_test)
    print('\n')

    results = [accuracy_score(y_test, y_keras_test), precision_score(y_test, y_keras_test,average='micro'), recall_score(y_test, y_keras_test,average='micro'), f1_score(y_test, y_keras_test,average='micro')]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"], columns={'gpu_logistic_regression'})
    
    return metrics_dataframe
    
    
    # Multi-layer perceptron with GPU
def gpu_mlp(X, X_train, X_test, y, y_train, y_test, cv, number_of_classes, gpu_mlp_activation, gpu_mlp_optimizer, gpu_mlp_epochs, gpu_mlp_loss, output_folder=None):
        
    """
    Multi-Layer perceptron using GPU

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        number_of_classes: the number of labels
        gpu_activation: The activation functions such as softmax, sigmoid, linear, relu or tanh.
        gpu_optimizer: The model optimizers such as stochastic gradient descent ('sgd'), adam ('adam') or RMSprop ('RMSprop').
        gpu_loss: The loss functions such as the mean squared error previously described ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic loss ('categorical_crossentropy').
        gpu_epochs: The number of epochs can also be choosen


    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.optimizers import SGD
    from sklearn.metrics import classification_report
    
    # Print number of GPUs available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # For future use, which devices the operations and tensors are assigned to (GPU, CPU)
    #tf.debugging.set_log_device_placement(True)
        
    # Define number of classes and number of features to include in our model
    number_of_classes = number_of_classes
    number_of_features = X_train.shape[1]

    keras_model = Sequential()
    keras_model.add(Dense(number_of_classes, activation=gpu_mlp_activation))
    keras_model.add(Flatten(input_dim=number_of_features))
    keras_model.compile(optimizer = gpu_mlp_optimizer,
                    loss = gpu_mlp_loss,
                    metrics = [gpu_mlp_loss])

    keras_model.fit(X_train, y_train, epochs=gpu_mlp_epochs)

    keras_model.evaluate(X_test, y_test) # loss, sparse_categorical_accuracy

    if output_folder is not None:
        keras_model.save(output_folder+'gpu_mlp')
        
    # Predicting X_test data with the created model
    y_keras_pred = keras_model.predict(X_test)
    y_keras_test = np.argmax(y_keras_pred,axis=1) #Make labels back
    
    
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        
    print("Classfication report for multi-layer perceptron using GPUs (if available)\n")
    print(classification_report(y_test, y_keras_test))
    print("\n")
                            
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', y_keras_test)
    print('\n')
    print('Real values: \n', y_test)
    print('\n')

    results = [accuracy_score(y_test, y_keras_test), precision_score(y_test, y_keras_test,average='micro'), recall_score(y_test, y_keras_test,average='micro'), f1_score(y_test, y_keras_test,average='micro')]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"], columns={'gpu_mlp'})
    
    return metrics_dataframe
    
def gpu_rnn(X, X_train, X_test, y, y_train, y_test, cv, rnn_units, rnn_activation, rnn_optimizer, rnn_loss, rnn_epochs, output_folder=None):
        
    """
    Recurrent Neural Network using GPUs if available

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        rnn_units: Positive integer, dimensionality of the output space.
        rnn_activation:  Activation function to use (softmax, sigmoid, linear, relu or tanh)
        rnn_optimizer: Optimizer (adam, sgd, RMSprop)
        rnn_loss: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic loss ('categorical_crossentropy').
        rnn_epochs: Number (Integer) of epochs

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score


    """

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
    from tensorflow.keras.optimizers import SGD
    from sklearn.metrics import classification_report

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
    
    model_lstm.evaluate(X_test, y_test) # loss, sparse_categorical_accuracy
    
    if output_folder is not None:
        model_lstm.save(output_folder+'gpu_rnn')

    # Predicting X_test data with the created model
    y_keras_pred = model_lstm.predict(X_test)
    y_keras_test = np.argmax(y_keras_pred,axis=1) #Make labels back
    
    
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        
    print("Classfication report for RNN using GPUs (if available)\n")
    print(classification_report(y_test, y_keras_test))
    print("\n")
                            
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', y_keras_test)
    print('\n')
    print('Real values: \n', y_test)
    print('\n')

    results = [accuracy_score(y_test, y_keras_test), precision_score(y_test, y_keras_test,average='micro'), recall_score(y_test, y_keras_test,average='micro'), f1_score(y_test, y_keras_test,average='micro')]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"], columns={'gpu_rnn'})
    
    return metrics_dataframe

def conv2d(X, X_train, X_test, y, y_train, y_test, conv_activation, conv_kernel_size, conv_optimizer, conv_loss, conv_epochs, output_folder=None):

    """
    2D Convolutional Neural Network using GPUs if available

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        conv_kernel_size: The kernel_size is the size of the filter matrix for the convolution (conv_kernel_size x conv_kernel_size).
        conv_activation:  Activation function to use (softmax, sigmoid, linear, relu or tanh)
        conv_optimizer: Optimizer (adam, sgd, RMSprop)
        conv_loss: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic loss ('categorical_crossentropy').
        conv_epochs: Number (Integer) of epochs

    Output:
       Accuracy of the model using (X_test and y_test)

    """

    from tensorflow.keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten

    # Creation of the model using Sequential() to build a model layer by layer
    model = Sequential()

    # Adding model layers
    # The first two layers are convolutional layers (Conv2D) dealing with 2D matrices
    # We set the first layer with X_train.shape[1]/1000 nodes (size of training data set / 1000) and the second layer with X_train.shape[1]/2000 nodes. We can adjust these numbers.
    # We choose an activation function for our first two layers, softmax for the last one.
    # We set the kernel_size parameter which means that the size of the filter matrix for the convolution is conv_kernel_size x conv_kernel_size.
    # The input_shape is simply the size of our images and 1 means that the image is greyscale
    model.add(Conv2D(X_train.shape[0]/1000, kernel_size=conv_kernel_size, activation=conv_activation, input_shape=(X_train.shape[1],X_train.shape[2],1)))
    model.add(Conv2D(X_train.shape[0]/2000, kernel_size=conv_kernel_size, activation=conv_activation))
    # Here we add a Flatten layer between the Convolutional layers and the Dense layer in order to connect both of them.
    model.add(Flatten())
    # The Dense layer is our output layer (standard) with the softmax activation function in order to make the output sum up to 1.
    # It means that we will have "probabilities" to predict our images
    model.add(Dense(X_test.shape[0]/1000, activation='softmax'))

    # Compile model
    # We use an optimizer and a loss function.
    # We use accuracy to measure model performance
    model.compile(optimizer=conv_optimizer, loss=conv_loss, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=conv_epochs)

    # Let's predict the first 10 images in the test set
    predictions = model.predict(X_test[:10])
    
    if output_folder is not None:
        model.save(output_folder+'conv2d')
        
    print("\n")
    print("Prediction of the first 10 images from the test dataset: ")
    print("\n")
    print(predictions)
    print("\n")

    # Actual results for first 5 images in test set
    print("Actual values of the first 10 images from the test dataset:")
    print("\n")
    print(y_test[:10])

    # Evaluate model performance on test data
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    return test_acc

