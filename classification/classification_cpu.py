#!/usr/bin/python3
# classification_cpu.py
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

"""

Classification algorithms used only with CPUs :
    - svm_linear,
    - svm_rbf,
    - svm_sigmoid,
    - svm_poly,
    - logistic_regression,
    - lda,
    - qda,
    - gnb,
    - mnb,
    - kneighbors,
    - sgd,
    - nearest_centroid,
    - decision_tree,
    - random_forest,
    - extra_trees
    - mlp_neural_network
    - mlp_neural_network_auto

"""

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier


from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

    
"""
    
SUPPORT VECTOR MACHINES
    
Support Vector Machine (SVM) is a supervised learning algorithm that can be used for both prediction of a binary variable (classification) and quantitative variable (regression
problems) even if it is primarily used for classification problems. The goal of SVM is to create a hyperplane that linearly divide n-dimensional data points in two components by
searching for an optimal margin that correctly segregate the data into different classes and at the same time be separated as much as possible from all the observations. In
addition to linear classification, it is also possible to compute a non-linear classification using what we call the kernel trick (kernel function) that maps inputs into high
dimensional feature spaces. The kernel function adapted to specific problems allows a real flexibility to adapt to different situations. SVM allows to create a classifier, or a
discrimination function, that we can generalize and apply for prediction such as in image classification, diagnostics, genomic sequences, or drug discovery.  SVM was developed at
AT&T Bell Laboratories by Vladimir Vapnik and colleagues. To select the optimal hyperplane amongst many hyperplanes that might classify our data, we select the one
that has the largest margin or in another words that represents the largest separation between the different classes. It is an optimization problem under constraints where the
distance between the nearest data point and the optimal hyperplane (on each side) is maximized. The hyperplane is then called the maximum-margin hyperplane allowing us to create a
maximum-margin classifier. The closest data-points are known as support vectors and margin is an area which generally do not contains any data points. If the optimal hyperplane is
too close to data points and the margin too small, it will be difficult to predict new data and the model will fail to generalize well. In non-linear cases, we will need to
introduce a kernel function to search for nonlinear separating surfaces. The method will induce a nonlinear transformation of our dataset towards an intermediate space that we
call feature space of higher dimension.
    
"""
    
"""
Support Vector Machines for classification
"""
# SVM with linear kernel
def svm_linear(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    SVM with linear kernel
    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model speparated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    #y_train = y_train.values.ravel()
    #y_test = y_test.values.ravel()
    
    model=svm.SVC(kernel='linear')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svm_linear.joblib')
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'SVM_linear'})

    print('Classification Report for SVM Linear\n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

# SVM with RBF kernel
def svm_rbf(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    SVM with RBF kernel
    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model speparated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    #y_train = y_train.values.ravel()
    #y_test = y_test.values.ravel()
    
    model=svm.SVC(kernel='rbf')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svm_rbf.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'SVM_rbf'})

    print('Classification Report for SVM RBF\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
# SVM with sigmoid kernel
def svm_sigmoid(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    SVM with sigmoid kernel
    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    
    #y_train = y_train.values.ravel()
    #y_test = y_test.values.ravel()
    
    model=svm.SVC(kernel='sigmoid')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svm_sigmoid.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'SVM_sigmoid'})

    print('Classification Report for SVM Sigmoid\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
# SVM with polynomial kernel
def svm_poly(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    SVM with polynomial kernel
    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
    """
    
    #y_train = y_train.values.ravel()
    #y_test = y_test.values.ravel()
    
    model=svm.SVC(kernel='poly')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'svm_poly.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'SVM_poly'})

    print('Classification Report for SVM Polynomial\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe

"""
Multinomial Logistic Regression
"""

# Multinomial Logistic Regression
def logistic_regression(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Multinomial Logistic Regression

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=LogisticRegression(solver='sag', multi_class='auto')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'logistic_regression.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'logistic_regression'})
    
    print('Classification Report for Logistic Regression\n')
    print(classification_report(y_test,y_pred))
    
    # Compute and print predicted output with X_test as new input data
    print('Print predicted output with X_test as new input data \n')
    print('\n')
    print('Predictions: \n', model.predict(X_test))
    print('\n')
    print('Real values: \n', y_test)
    print('\n')
        
    return metrics_dataframe


"""
Linear Discriminant Analysis
    
LDA is a supervised machine learning technique used to classify data. It is also used as a dimensionality reduction technique to project the features from a higher dimension space
into a lower dimension space with good class-separability avoiding overfitting and reduce computational costs. PCA and LDA are comparable in the sense that they are linear
transformations. The first one is an unsupervised algorithm that will find the principal components, the directions, that maximize the variance in the data set, and it ignores the
class labels. LDA is not an unsupervised algorithm as it considers the class labels, supervised algorithm, and computes the linear discriminants, the directions, that maximize the
separation between multiple classes. In other words, LDA will maximize the distance between the mean of each class and minimize the spreading within the class itself (minimizing
variation between each category).

"""
    
# Linear discriminant analysis
def lda(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Linear Discriminant Analysis

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=LinearDiscriminantAnalysis()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'lda.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'lda'})

    print('Classification Report for LDA\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
# Quadratic discriminant analysis
def qda(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Quandratic Discriminant Analysis

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=QuadraticDiscriminantAnalysis()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'qda.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'qda'})

    print('Classification Report for QDA\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe

"""
Naive Bayes
"""

# Gaussian Naive Bayes estimator
def gnb(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Gaussian Naive Bayes estimator

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'gnb.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'gnb'})

    print('Classification Report for GNB\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
# Multinomial Naive Bayes estimator
def mnb(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Multinomial Naive Bayes estimator

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=MultinomialNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'mnb.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'mnb'})

    print('Classification Report for MNB\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe


    
"""
K-Neighbors
"""

# K-Neighbors
def kneighbors(X, X_train, X_test, y, y_train, y_test, cv, n_neighbors, output_folder = None):
    """
    K-Neighbors Naive Bayes estimator

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        n_neighbors: Number of neighbors

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'kneighbors.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'kneighbors'})

    print('Classification Report for K-Neighbors\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe


"""
Stochastic Gradient Descent
"""

# Stochastic Gradient Descent
def sgd(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Stochastic Gradient Descent

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=SGDClassifier(loss="hinge", penalty="l2")
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'sgd.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'sgd'})

    print('Classification Report for SGD\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe


"""
Nearest Centroid Classifier
"""

# Stochastic Gradient Descent
def nearest_centroid(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    K-Nearest Centroid

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=NearestCentroid()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'nearest_centroid.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'nearest_centroid'})

    print('Classification Report for Nearest Centroid\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe

"""
Trees
"""

# Decision Tree
def decision_tree(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Decision Tree

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=tree.DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'decision_tree.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'decision_tree'})

    print('Classification Report for Decision Tree\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
   
# Random Forest
def random_forest(X, X_train, X_test, y, y_train, y_test, cv, n_estimators = None, output_folder = None):
    """
    Random Forest

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=RandomForestClassifier(n_estimators)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'random_forest.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'random_forest'})

    print('Classification Report for Random Forest\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe

# Extremely Randomized Trees
def extra_trees(X, X_train, X_test, y, y_train, y_test, cv, n_estimators = None, output_folder = None):
    """
    Extremely Randomized Trees

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation selected in inputs.py
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation (defined in inputs.py)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """

    model=ExtraTreesClassifier(n_estimators)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'extra_trees.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'extra_trees'})

    print('Classification Report for Extra Trees\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe

def mlp_neural_network(X, X_train, X_test, y, y_train, y_test, cv, max_iter = None, hidden_layer_sizes = None, mlp_activation = None, solver = None, alpha = None, mlp_learning_rate = None, learning_rate_init = None, output_folder = None):

    """
    Multi-layer Perceptron classifier

    Inputs:
        X,y non splitted dataset separated by features (X) and labels (y).
        X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
        X_test, y_test: selected dataset to test the model separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation 
        max_iter: Maximum number of iterations (default= 200)
        hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
        mlp_activation: Activation function for the hidden layer ('identity', 'logistic', 'relu', 'softmax', 'tanh'). default=’relu’
        solver: The solver for weight optimization (‘lbfgs’, ‘sgd’, ‘adam’). default=’adam’
        alpha: Strength of the L2 regularization term (default=0.0001)
        mlp_learning_rate: Learning rate schedule for weight updates (‘constant’, ‘invscaling’, ‘adaptive’). default='constant'
        learning_rate_init: The initial learning rate used (for sgd or adam). It controls the step-size in updating the weights.
        
    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score
            
    """
    model = MLPClassifier(max_iter = max_iter, hidden_layer_sizes = hidden_layer_sizes, activation = mlp_activation, solver = solver, alpha = alpha, learning_rate = mlp_learning_rate, learning_rate_init = learning_rate_init)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'mlp_neural_network.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'MLP_neural_network'})

    print('Classification Report for MLP Neural Network:\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
    
# Neural Network (MLP)
def mlp_neural_network_auto(X, X_train, X_test, y, y_train, y_test, cv, output_folder = None):
    """
    Multi-layer Perceptron classifier: Run automatically different hyperparameters combinations and return the best result

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
    mlp_gs = MLPClassifier()
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
    y_pred = model.predict(X_test)

    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder+'mlp_neural_network_auto.joblib')
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns={'mlp_neural_network_auto'})

    print('Classification Report for MLP Neural Network Auto:\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
    

