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

# Import necessary libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Configure warning settings
warnings.filterwarnings('always')  # Options: "error", "ignore", "always", "default", "module", or "once"

    
"""
SUPPORT VECTOR MACHINES

A Support Vector Machine (SVM) is a supervised learning algorithm primarily utilized for classification tasks but can also be applied to regression problems. The objective of SVM is to create a hyperplane that linearly separates n-dimensional data points into two distinct groups while maximizing the margin between these groups. The margin represents the distance between the hyperplane and the nearest data points, known as support vectors, and it aims to be as large as possible to improve the generalization and prediction capabilities of the model.

In cases where linear separation is not possible, SVM employs the kernel trick to map inputs into higher-dimensional feature spaces, enabling non-linear classification. By adapting kernel functions to specific problems, SVM offers flexibility in handling diverse situations. As a result, SVM can be used to create classifiers or discrimination functions for various applications, including image classification, diagnostics, genomic sequences, and drug discovery.

Developed by Vladimir Vapnik and colleagues at AT&T Bell Laboratories, SVM selects the optimal hyperplane among multiple candidates by maximizing the margin, which ensures the largest separation between classes. This process is essentially an optimization problem with constraints. The resulting maximum-margin hyperplane creates a maximum-margin classifier that generalizes well when applied to new data.

In non-linear scenarios, a kernel function is introduced to identify non-linear separating surfaces. This method applies a non-linear transformation to the dataset, projecting it into a higher-dimensional feature space.
"""
    
"""
Support Vector Machines for classification
"""

# Import necessary libraries
from joblib import dump, load
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def svm_linear(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    SVM with linear kernel
    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """
    
    # Create and train SVM model with linear kernel
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'svm_linear.joblib')

    # Display predicted and actual labels
    print("\nPredicted labels for X_test:")
    print(y_pred)
    print("\nActual labels:")
    print(y_test)
    print("\n")
    
    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]
    
    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'SVM_linear'})

    # Display classification report
    print('Classification Report for SVM Linear\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe


"""
Support Vector Machines for classification with RBF kernel
"""

# Import necessary libraries
from joblib import dump, load
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def svm_rbf(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    SVM with RBF kernel
    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train SVM model with RBF kernel
    model = svm.SVC(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'svm_rbf.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'SVM_rbf'})

    # Display classification report
    print('Classification Report for SVM RBF\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe
    
"""
Support Vector Machines for classification with sigmoid kernel
"""

# Import necessary libraries
from joblib import dump, load
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def svm_sigmoid(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    SVM with sigmoid kernel
    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train SVM model with sigmoid kernel
    model = svm.SVC(kernel='sigmoid')
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'svm_sigmoid.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'SVM_sigmoid'})

    # Display classification report
    print('Classification Report for SVM Sigmoid\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe
    
"""
Support Vector Machines for classification with polynomial kernel
"""

# Import necessary libraries
from joblib import dump, load
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def svm_poly(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    SVM with polynomial kernel
    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train SVM model with polynomial kernel
    model = svm.SVC(kernel='poly')
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'svm_poly.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'SVM_poly'})

    # Display classification report
    print('Classification Report for SVM Polynomial\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe


"""
Multinomial Logistic Regression
"""

# Import necessary libraries
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def logistic_regression(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Multinomial Logistic Regression

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a multinomial logistic regression model
    model = LogisticRegression(solver='sag', multi_class='auto')
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'logistic_regression.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'logistic_regression'})

    # Display classification report
    print('Classification Report for Logistic Regression\n')
    print(classification_report(y_test, y_pred))

    # Print predicted and real output values
    print('Print predicted output with X_test as new input data \n')
    print('Predictions: \n', y_pred)
    print('Real values: \n', y_test)

    return metrics_dataframe



"""
Linear Discriminant Analysis (LDA)

LDA is a supervised machine learning technique used for classification and dimensionality reduction. It projects features from a higher-dimensional space into a lower-dimensional
space with good class-separability, avoiding overfitting and reducing computational costs. Unlike PCA, which is an unsupervised algorithm that maximizes variance, LDA is a
supervised algorithm that maximizes the separation between multiple classes. LDA aims to maximize the distance between class means and minimize the spreading within each class.
"""

# Import necessary libraries
from joblib import dump, load
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def lda(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Linear Discriminant Analysis

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train an LDA model
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'lda.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'lda'})

    # Display classification report
    print('Classification Report for LDA\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe

    
"""
Quadratic Discriminant Analysis (QDA)

QDA is a supervised machine learning technique used for classification. It is an extension of Linear Discriminant Analysis (LDA) and assumes that each class has its own
covariance matrix, allowing for more flexibility in modeling the data. This can lead to improved performance in cases where the classes are not linearly separable.
"""

# Import necessary libraries
from joblib import dump, load
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def qda(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Quadratic Discriminant Analysis

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a QDA model
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'qda.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'qda'})

    # Display classification report
    print('Classification Report for QDA\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe


"""
Naive Bayes

Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between every pair of features given the value of the class variable.
"""

# Import necessary libraries
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd

def gnb(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Gaussian Naive Bayes estimator

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'gnb.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'gnb'})

    # Display classification report
    print('Classification Report for GNB\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe
    
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from joblib import dump, load

def mnb(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Multinomial Naive Bayes estimator

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'mnb.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'mnb'})

    # Display classification report
    print('Classification Report for MNB\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from joblib import dump, load

def kneighbors(X, X_train, X_test, y, y_train, y_test, cv, n_neighbors, output_folder=None):
    """
    K-Neighbors classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        n_neighbors: Number of neighbors

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a K-Neighbors model with the specified number of neighbors
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'kneighbors.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'kneighbors'})

    # Display classification report
    print('Classification Report for K-Neighbors\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe



from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from joblib import dump, load

def sgd(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Stochastic Gradient Descent classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train an SGD model with hinge loss and L2 penalty
    model = SGDClassifier(loss="hinge", penalty="l2")
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'sgd.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'sgd'})

    # Display classification report
    print('Classification Report for SGD\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe

from sklearn.neighbors import NearestCentroid
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from joblib import dump, load

def nearest_centroid(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Nearest Centroid Classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a Nearest Centroid model
    model = NearestCentroid()
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'nearest_centroid.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'nearest_centroid'})

    # Display classification report
    print('Classification Report for Nearest Centroid\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe



from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import classification_report
from joblib import dump, load

def decision_tree(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Decision Tree Classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'decision_tree.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'decision_tree'})

    # Display classification report
    print('Classification Report for Decision Tree\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe

   
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import classification_report
from joblib import dump, load

def random_forest(X, X_train, X_test, y, y_train, y_test, cv, n_estimators=None, output_folder=None):
    """
    Random Forest Classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        n_estimators: number of trees in the forest (default=None)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        dump(model, output_folder + 'random_forest.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'random_forest'})

    # Display classification report
    print('Classification Report for Random Forest\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe


from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import classification_report

def extra_trees(X, X_train, X_test, y, y_train, y_test, cv, n_estimators=None, output_folder=None):
    """
    Extremely Randomized Trees Classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y). Used for cross-validation
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        n_estimators: number of trees in the forest (default=None)

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train an Extra Trees model
    model = ExtraTreesClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder + 'extra_trees.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'extra_trees'})

    # Display classification report
    print('Classification Report for Extra Trees\n')
    print(classification_report(y_test, y_pred))

    return metrics_dataframe

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import classification_report

def mlp_neural_network(X, X_train, X_test, y, y_train, y_test, cv, max_iter=None, hidden_layer_sizes=None, mlp_activation=None, solver=None, alpha=None, mlp_learning_rate=None, learning_rate_init=None, output_folder=None):
    """
    Multi-layer Perceptron Classifier

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y)
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation
        max_iter: maximum number of iterations (default=200)
        hidden_layer_sizes: ith element represents the number of neurons in the ith hidden layer
        mlp_activation: activation function for the hidden layer ('identity', 'logistic', 'relu', 'softmax', 'tanh'), default='relu'
        solver: solver for weight optimization ('lbfgs', 'sgd', 'adam'), default='adam'
        alpha: strength of the L2 regularization term, default=0.0001
        mlp_learning_rate: learning rate schedule for weight updates ('constant', 'invscaling', 'adaptive'), default='constant'
        learning_rate_init: initial learning rate used (for sgd or adam), controls the step-size in updating the weights

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Create and train a MLP model
    model = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, activation=mlp_activation, solver=solver, alpha=alpha, learning_rate=mlp_learning_rate, learning_rate_init=learning_rate_init)
    model.fit(X_train, y_train)

    # Predict labels for X_test
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder + 'mlp_neural_network.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'MLP_neural_network'})

    # Display classification report
    print('Classification Report for MLP Neural Network:\n')
    print(classification_report(y_test,y_pred))
    
    return metrics_dataframe
    
    
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
from sklearn.metrics import classification_report

def mlp_neural_network_auto(X, X_train, X_test, y, y_train, y_test, cv, output_folder=None):
    """
    Multi-layer Perceptron classifier: Run automatically different hyperparameters combinations and return the best result

    Inputs:
        X, y: non-splitted dataset separated by features (X) and labels (y)
        X_train, y_train: dataset to train the model, separated by features (X_train) and labels (y_train)
        X_test, y_test: dataset to test the model, separated by features (X_test) and labels (y_test)
        cv: number of k-folds for cross-validation

    Output:
        A DataFrame with the following metrics:
            accuracy_score: ratio of the number of correct predictions to all number predictions made by the classifiers
            precision_score: number of correct outputs or how many of the correctly predicted cases turned out to be positive
            recall_score: how many of the actual positive cases we were able to predict correctly
            f1_score: harmonic mean of precision and recall
            cross_val_score: Cross-validation score
    """

    # Define estimator and hyperparameter search space
    mlp_gs = MLPClassifier()
    parameter_space = {
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.01, 0.001, 0.05, 0.005],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
    }

    # Perform hyperparameter search using GridSearchCV
    model = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    model.fit(X_train, y_train)

    # Make predictions using the best model
    y_pred = model.predict(X_test)

    # Save the model to a file if output_folder is specified
    if output_folder is not None:
        from joblib import dump, load
        dump(model, output_folder + 'mlp_neural_network_auto.joblib')

    # Calculate metrics and cross-validation scores
    results = [
        metrics.accuracy_score(y_test, y_pred),
        metrics.precision_score(y_test, y_pred, average='micro'),
        metrics.recall_score(y_test, y_pred, average='micro'),
        metrics.f1_score(y_test, y_pred, average='micro'),
        cross_val_score(model, X_train, y_train, cv=cv).mean(),
        cross_val_score(model, X_train, y_train, cv=cv).std()
    ]

    # Create a DataFrame with calculated metrics
    metrics_dataframe = pd.DataFrame(results,
                                     index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"],
                                     columns={'mlp_neural_network_auto'})

    # Display classification report and best hyperparameters
    print('Classification Report for MLP Neural Network Auto:\n')
    print(classification_report(y_test, y_pred))
    print('Best parameters found:\n', model.best_params_)

    return metrics_dataframe

    

