#!/usr/bin/python3
# datasets.py
# Author: Xavier Vasques (Last update: 28/05/2022)

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

import pandas as pd


# bottle.csv
def bottle():
    """
    
    California Cooperative Oceanic Fisheries Investigations (CalCOFI) data set representing the longest (1949-present) and most complete (more than 50,000 sampling stations) time series
    of oceanographic and larval fish data in the world (https://www.kaggle.com/datasets/sohier/calcofi?select=bottle.csv). This dataset became valuable for documenting climatic cycles
    locally. We extract the following variables:
            - Depthm: depth in meters
            - T_degC: water temperature in degree Celsius that we rename as "Target"
            - Salnty: salinity in g of salt per kg of water (g/kg)
            - O2ml_L: O2 mixing ration in ml/L
    The data is great to test regression algorithms
    
    Output: A pandas DataFrame
    
    """
    
    # Load csv file, read and create a pandas DataFrame
    csv_data = './data/datasets/bottle.csv'
    df = pd.read_csv(csv_data, delimiter=',')
    # Select the variables we want to keep: 'Depthm','T_degC','Salnty','O2ml_L'
    df = df[['Depthm','T_degC','Salnty','O2ml_L']]
    # Drop row having at least 1 missing value
    df = df.dropna()
    # Rename "T_degC" feature to "Target"
    df = df.rename(columns={"T_degC": "Target"})
    return df

# neurons.csv
def neurons():
    """
    
    We extract the features from 29321 labeled rat neurons from neuromorpho.org. The data contains two main classes (principal cell and interneuron cell) and 14 subclasses, including six
    types of principal cells (ganglion, granule, medium spiny, parachromaffin, Purkinje, and pyramidal cells), six types of interneurons (basket, chandelier, martinotti, double bouquet,
    bitufted, nitrergic) and two types of glial cells (microglia and astrocytes).
    
    The data is great to test classification algorithms
    
    Output: A pandas DataFrame
    
    """
    
    # Load csv file, read and create a pandas DataFrame
    data = './data/datasets/neurons.csv'
    df = pd.read_csv(data, delimiter=';')
    # Drop row having at least 1 missing value
    df = df.dropna()
    return df

# neurons_mahalanobis.csv
def neurons_mahalanobis():
    """
    
    We extracted the features from 29321 labeled rat neurons from neuromorpho.org. We applied the mahalanobis methodoloy and get 2628 neuron morphologies. The data contains two main classes (principal cell and interneuron cell) and 14 subclasses, including six
    types of principal cells (ganglion, granule, medium spiny, parachromaffin, Purkinje, and pyramidal cells), six types of interneurons (basket, chandelier, martinotti, double bouquet,
    bitufted, nitrergic) and two types of glial cells (microglia and astrocytes).
        
    Output: A pandas DataFrame
    
    """
    
    # Load csv file, read and create a pandas DataFrame
    data = './data/datasets/neurons_mahalanobis.csv'
    df = pd.read_csv(data, delimiter=',')
    # Drop row having at least 1 missing value
    df = df.dropna()

    return df
    
# neurons_maha_soma.csv
def neurons_maha_soma():
    """
    
    We extracted the features from 29321 labeled rat neurons from neuromorpho.org. We delete rows with soma surface = 0 and applied the mahalanobis methodoloy and get 2628 neuron morphologies. The data contains two main classes (principal cell and interneuron cell) and 14 subclasses, including six
    types of principal cells (ganglion, granule, medium spiny, parachromaffin, Purkinje, and pyramidal cells), six types of interneurons (basket, chandelier, martinotti, double bouquet,
    bitufted, nitrergic) and two types of glial cells (microglia and astrocytes).
        
    Output: A pandas DataFrame
    
    """
    
    # Load csv file, read and create a pandas DataFrame
    data = './data/datasets/neurons_maha_soma.csv'
    df = pd.read_csv(data, delimiter=',')
    # Drop row having at least 1 missing value
    df = df.dropna()

    return df


    
# neurons_binary.csv
def neurons_binary():
    """
    
    We extract the features from 29321 labeled rat neurons from neuromorpho.org. The data contains two main classes (principal cell and interneuron cell).
    
    The data is great to test classification algorithms (binary)
    
    Output: A pandas DataFrame
    
    """
    
    # Load csv file, read and create a pandas DataFrame
    data = './data/datasets/neurons_binary.csv'
    df = pd.read_csv(data, delimiter=';')
    # Drop row having at least 1 missing value
    df = df.dropna()
    return df
    
# fashion-mnist_train.csv
def fashion_mnist_train():
    """
    Famous Fashion MNIST dataset which is a MNIST-like data set of 70000 28x28 labeled Zalando’s article images (784 pixels in total per image). We have a training set of 60 000 examples
    that we will use here. It is also available a test set of 10000 examples.  We can find the dataset in Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist. Each
    pixel has a single value between 0 and 255 indicating the lightness of that pixel. Highest the value, darker the pixel. When the data is extracted, we have 785 columns with the
    labels (0: T-shirt/top, 1: Trouser, 2: Pullover etc.) and the rest of the columns the 784 features which are the pixel numbers and their respective values).
    
    Output: A pandas DataFrame and Labels
        
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    csv_data = './data/datasets/fashion-mnist_train.csv'
    df = pd.read_csv(csv_data, delimiter=',')
    
    # Rename "label" feature to "Target"
    df = df.rename(columns={"label": "Target"})
    
    # Initiate the label values
    Labels = {0:'T-shirt/top',
            1:'Trouser',
            2:'Pullover',
            3:'Dress',
            4:'Coat',
            5:'Sandal',
            6:'Shirt',
            7:'Sneaker',
            8:'Bag',
            9:'Ankle boot'}
          
    return df, Labels
    
# fashion-mnist_test.csv
def fashion_mnist_test():
    """
    Famous Fashion MNIST dataset which is a MNIST-like data set of 70000 28x28 labeled Zalando’s article images (784 pixels in total per image). We have a training set of 60 000 examples
    that we will use here. It is also available a test set of 10000 examples.  We can find the dataset in Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist. Each
    pixel has a single value between 0 and 255 indicating the lightness of that pixel. Highest the value, darker the pixel. When the data is extracted, we have 785 columns with the
    labels (0: T-shirt/top, 1: Trouser, 2: Pullover etc.) and the rest of the columns the 784 features which are the pixel numbers and their respective values).
    
    Output: A pandas DataFrame and Labels
    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    csv_data = './data/datasets/fashion-mnist_test.csv'
    df = pd.read_csv(csv_data, delimiter=',')
    
    # Rename "label" feature to "Target"
    df = df.rename(columns={"label": "Target"})
    
    # Initiate the label values
    Labels = {0:'T-shirt/top',
            1:'Trouser',
            2:'Pullover',
            3:'Dress',
            4:'Coat',
            5:'Sandal',
            6:'Shirt',
            7:'Sneaker',
            8:'Bag',
            9:'Ankle boot'}
          
    return df, Labels
    
# breastcancer.csv
def breastcancer():
    """
    
    UCI ML Breast Cancer Wisconsin (Diagnostic) dataset. It is composed of two classes (WDBC-Malignant, WDBC-Benign), 30 numeric attributes, with 569 samples (212 for malignant and 357
    for benign). The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe characteristics of the cell nuclei present in the image
    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    breastcancer = './data/datasets/breastcancer.csv'
    df = pd.read_csv(breastcancer, delimiter=';')

    # Rename "diagnosis" feature to "Target"
    df = df.rename(columns={"diagnosis": "Target"})

    return df

# creditcard.csv
def creditcard():
    """
    
    Fraudulent credit card transactions (https://www.kaggle.com/mlg-ulb/creditcardfraud).

    The dataset contains transaction made by credit cards in September 2013 by European cardholders during two days in which we can find 492 frauds (0.172%) out of 284,807 transactions.
    The data was transformed by using PCA. Principal components (V1, V2, V3, ….) are the new features except for “Time” and “Amount” which are the original ones. The feature “Class” is
    the response variable with the value 1 in case of fraud and 0 otherwise.

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    creditcard = './data/datasets/creditcard.csv'
    df = pd.read_csv(creditcard, delimiter=',')
    
    # Rename "Class" feature to "Target"
    df = df.rename(columns={"Class": "Target"})
    
    return df


# DailyDelhiClimateTrain.csv
def DailyDelhiClimateTrain():
    """
    
    Weather forecasting for Indian climate. This dataset provides data from 2013 to 2017 in the city of Delhi, India with four parameters: meantemp, humidity, wind_speed, meanpressure

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    DailyDelhiClimateTrain = './data/datasets/DailyDelhiClimateTrain.csv'
    df = pd.read_csv(DailyDelhiClimateTrain, delimiter=',')
    
    return df

# DailyDelhiClimateTest.csv
def DailyDelhiClimateTest():
    """
    
    Weather forecasting for Indian climate. This dataset provides data from 2013 to 2017 in the city of Delhi, India with four parameters: meantemp, humidity, wind_speed, meanpressure

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    DailyDelhiClimateTest = './data/datasets/DailyDelhiClimateTest.csv'
    df = pd.read_csv(DailyDelhiClimateTest, delimiter=',')
    
    return df

# brain_train.csv
def brain_train():
    """
    
    Extration of features from brain MRIs

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    breastcancer = './data/datasets/brain_train.csv'
    df = pd.read_csv(breastcancer, delimiter=';')
    
    return df
    
# brain_test.csv
def brain_test():
    """
    
    Extration of features from brain MRIs

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    breastcancer = './data/datasets/brain_test.csv'
    df = pd.read_csv(breastcancer, delimiter=';')
    
    return df

# diabetes.csv
def diabetes():
    """
    
    National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to predict whether a patient has diabetes based on several measurements
    (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    diabetes = './data/datasets/diabetes.csv'
    df = pd.read_csv(diabetes, delimiter=',')

    # Rename "Outcome" feature to "Target"
    df = df.rename(columns={"Outcome": "Target"})
    
    return df

# insurance.csv
def insurance():
    """
    
    Medical cost personal dataset (https://www.kaggle.com/mirichoi0218/insurance) which is used for insurance forecast by using linear regression. In the dataset, we will find costs
    billed by health insurance companies (insurance charges) and features (age, gender, BMI, children, smoker).

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    insurance = './data/datasets/insurance.csv'
    df = pd.read_csv(insurance, delimiter=',')
    
    return df

# mushrooms.csv
def mushrooms():
    """
    
    Mushroom classification data set coming from Kaggle (https://www.kaggle.com/uciml/mushroom-classification): « This dataset includes descriptions of hypothetical samples corresponding
    to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is
    identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states
    that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy ».

    
    """

    # load data: Capture the dataset in Python using Pandas DataFrame
    mushrooms = './data/datasets/mushrooms.csv'
    df = pd.read_csv(mushrooms, delimiter=',')

    # Rename "Outcome" feature to "Target"
    df = df.rename(columns={"Class": "Target"})
    
    return df

# politics.csv

def politics():

    # load data: Capture the dataset in Python using Pandas DataFrame
    politics = './data/datasets/politics.csv'
    df = pd.read_csv(politics, delimiter=',')
    
    return df

def SMSSpamCollection():

    # load data: Capture the dataset in Python using Pandas DataFrame
    df = pd.read_csv('../data/datasets/SMSSpamCollection', sep='\t', names= ['label', 'message'])

    return df
