
# Import the required modules
from ml_pipeline_function import ml_pipeline_function
import pandas as pd
from keras.datasets import mnist

# Load the MNIST dataset and split it into training and test sets
(X, y), (_, _) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_test)

# Reshape the training and test data to fit the model
# The data has X_train.shape[0] images for training,
# image size is X_train.shape[1] x X_train.shape[2], and 1 means the image is grayscale.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Run the Machine Learning (ML) pipeline function with the following parameters:
# - Input data: X
# - Target values: y
# - Training data: X_train, y_train
# - Test data: X_test, y_test
# - Output folder: './Outputs/'
# - Convolutional layer type: 'conv2d'
# - Activation function: 'relu'
# - Kernel size: 3
# - Optimizer: 'adam'
# - Loss function: 'categorical_crossentropy'
# - Number of training epochs: 1
ml_pipeline_function(
    X, y, X_train, y_train, X_test, y_test,
    output_folder='./Outputs/',
    convolutional=['conv2d'],
    conv_activation='relu',
    conv_kernel_size=3,
    conv_optimizer='adam',
    conv_loss='categorical_crossentropy',
    conv_epochs=1
)
