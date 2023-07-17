#!/usr/bin/python3
# classification_qpu.py
# Author: Xavier Vasques (Last update: 11/06/2023)

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

# Importing utilities for data processing and visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce

# Importing scikit-learn utilities for classification and evaluation
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Importing qiskit_machine_learning utilities for classification and evaluation
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
#from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
#from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, ADAM, SPSA, AQGD, GradientDescent
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
#from qiskit_machine_learning.exceptions import QiskitMachineLearningError


"""
Quantum Algorithms for Classification.
    Part 1: Quantum Kernel Algorithms for Classification
    Part 2: Quantum Neural Networks for Classification
"""

"""
Part 1: Quantum Kernel Algorithms for classification

Encoding functions:
    - See papers from Havlíček et al. (q_kernel_zz), Glick et al. (q_kernel_training) and Suzuki et al. (q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_default)
        Havlíček, V. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).
        Glick, J. R. et al. Covariant quantum kernels for data with group structure. (2021) doi:10.48550/ARXIV.2105.03406.
        Suzuki, Y. et al. Analysis and synthesis of feature map for kernel-based quantum classifier. Quantum Mach. Intell. 2, 9 (2020).

- Inputs:
    - X, y: non-splitted dataset (used for cross-validation)
    - X_train, y_train: dataset for model training
    - X_test, y_test: dataset for model testing
    - cv: number of k-folds for cross-validation
    - feature_dimension: dimensionality of the data (equal to the number of required qubits)
    - reps: number of times the feature map circuit is repeated
    - ibm_account: IBM account (API)
    - multiclass: OneVsRestClassifier, OneVsOneClassifier, or svc
    - output_folder: path to save outputs
    - quantum_backend: backend options. If quantum_backend is None, then local simulator is used to compute (Aer simulators). Online simulators or hardware can be selected on https://www.ibm.com/quantum.
    - For Pegasos algorithms:
        - num_steps: number of steps in the training procedure
        - C: regularization parameter

- Output:
    - DataFrame with the following metrics:
        - accuracy_score: ratio of correct predictions to total predictions
        - precision_score: number of correct positive predictions
        - recall_score: number of actual positive cases predicted correctly
        - f1_score: harmonic mean of precision and recall
        - cross_val_score: cross-validation score
"""

def q_kernel_zz(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):
   
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
        
    # Quantum Feature Mapping with feature_dimension, reps and type of entanglements (linear, full, ...)
    from qiskit.circuit.library import ZZFeatureMap
    qfm_zz = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="linear")
    
    if quantum_backend is not None:
        # Compute code with online quantum simulators or quantum hardware from the cloud
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler with diferent options
        # resilience_level=1 adds readout error mitigation
        # execution.shots is the number of shots
        # optimization_level=3 adds dynamical decoupling
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1024
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        # Compute code with local simulator (Aer simulators)
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    # After preparing our training and testing datasets, we configure the FidelityQuantumKernel class to compute a kernel matrix using the ZZFeatureMap.
    # We utilize the default implementation of the Sampler primitive and the ComputeUncompute fidelity, which calculates the overlaps between states.
    # If you do not provide specific instances of Sampler or Fidelity, the code will automatically create these objects with the default values.
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_zz = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_zz)
    
    # multiclass option choosen
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_zz.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_zz.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_zz.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_zz)

    # Fit the model to the training data
    model.fit(X_train,y_train)
    # Evaluate the model's performance on the test data
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for q_kernel_zz: {score}')
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_zz.model")
    
    print("\n")
    print("Print predicted data (predicted labels) coming from X_test\n")
    print(y_pred)
    print("\n")
    print("Print real values (y_test)\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    for indices_train, indices_test in k_fold.split(X_train):
        # Split the training data into train and validation sets for each fold
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
        # Fit classifier to the training data for the current fold
        model.fit(X_train_, y_train_)
        # Score the classifier on the validation data for the current fold
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    # Calculate the mean, variance, and standard deviation of the cross-validation scores
    import math
    print("cross validation scores: \n", score)
    cross_mean = sum(score) / len(score) # mean
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: \n", cross_mean)

    # Calculate various metrics using sklearn.metrics and store the results in a dataframe
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_zz'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe
    
def q_kernel_training(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):
 
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed

    # normalize the data between 0 and 2pi
    X_train -= X_train.min(0)
    X_train /= X_train.max(0)
    X_train *= 2*np.pi

    # normalize the data between 0 and 2pi
    X_test -= X_test.min(0)
    X_test /= X_test.max(0)
    X_test *= 2*np.pi
    
    class QKTCallback:
        """Callback wrapper class."""

        def __init__(self) -> None:
            self._data = [[] for i in range(5)]

        def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
            """
            Args:
                x0: number of function evaluations
                x1: the parameters
                x2: the function value
                x3: the stepsize
                x4: whether the step was accepted
            """
            self._data[0].append(x0)
            self._data[1].append(x1)
            self._data[2].append(x2)
            self._data[3].append(x3)
            self._data[4].append(x4)

        def get_callback_data(self):
            return self._data

        def clear_callback_data(self):
            self._data = [[] for i in range(5)]
        

    # Qiskit imports
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.visualization import circuit_drawer
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
    from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
        
    # Create a rotational layer to train. We will rotate each qubit the same amount.
    training_params = ParameterVector("θ", 1)
    fm0 = QuantumCircuit(feature_dimension)
    for qubit in range(feature_dimension):
        fm0.ry(training_params[0], qubit)
        
    #fm0.ry(training_params[0], 0)
    #fm0.ry(training_params[0], 1)

    # Use ZZFeatureMap to represent input data
    fm1 = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement='linear')

    # Create the feature map, composed of our two circuits
    fm = fm0.compose(fm1)

    print(circuit_drawer(fm))
    print(f"Trainable parameters: {training_params}")
    
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    
    # Instantiate quantum kernel
    quant_kernel = TrainableFidelityQuantumKernel(fidelity = fidelity, feature_map=fm, training_parameters=training_params)

    # Set up the optimizer
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)

    # Instantiate a quantum kernel trainer.
    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel, loss="svc_loss", optimizer=spsa_opt, initial_point=[np.pi / 2]
    )
    
    # Train the kernel using QKT directly
    qka_results = qkt.fit(X_train, y_train)
    optimized_kernel = qka_results.quantum_kernel
    print(qka_results)
    
    # Use QSVC for classification
    qsvc = QSVC(quantum_kernel=optimized_kernel)

    # Fit the QSVC
    qsvc.fit(X_train, y_train)

    # Predict the labels
    labels_test = qsvc.predict(X_test)

    # Evalaute the test accuracy
    accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=labels_test)
    print(f"accuracy test: {accuracy_test}")
    
    if output_folder is not None:
        if multiclass is None:
            qsvc.save(output_folder+"qsvc.model")

    # Print predicted values and real values of the X_test dataset
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(labels_test)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        qsvc.fit(X_train_, y_train_)

        # score classifier
        score[i] = qsvc.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, labels_test),metrics.precision_score(y_test, labels_test, average='micro'),metrics.recall_score(y_test, labels_test, average='micro'),metrics.f1_score(y_test, labels_test, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_training'])
    print('Classification Report: \n')
    print(classification_report(y_test,labels_test))
            
    return metrics_dataframe
    


def q_kernel_default(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_default = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full')
    print(qfm_default)

    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_default = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_default)
    
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_default.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_default.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_default.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_default)
                
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_default: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_default.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_default'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def data_map_8(x: np.ndarray) -> float:
    """
    Define a function map from R^n to R.

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: np.pi*(m * n), x)
    return coeff

def data_map_9(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: (np.pi/2)*(m * n), 1 - x)
    return coeff

def data_map_10(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: np.pi*np.exp(((n - m)*(n - m))/8), x)
    return coeff

def data_map_11(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: (np.pi/3)*(m * n), 1/(np.cos(x)))
    return coeff

def data_map_12(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: np.pi*(m * n), np.cos(x))
    return coeff

def q_kernel_8(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed



    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_8 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_8)
    print(qfm_8)

    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_8 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_8)
    
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_8.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_8.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_8.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_8)
                
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_8: {score}')

    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_8.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")

    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
        
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_8'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_9(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_9 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_9)
    print(qfm_9)

    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_9 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_9)
    
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_9.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_9.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_9.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_9)
                
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_9: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_9.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
        
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_9'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_10(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_10 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_10)
    print(qfm_10)

    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_10 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_10)
    
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_10.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_10.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_10.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_10)
                
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_10: {score}')
    
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_10.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_10'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe


def q_kernel_11(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_11 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_11)
    print(qfm_11)

    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_11 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_11)
    
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_11.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_11.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_11.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_11)
                
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_11: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_11.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_11'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_12(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_12 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_12)
    print(qfm_12)
    
    # For quantum access, the following lines must be adapted
    
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_12 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_12)
    
    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=Q_Kernel_12.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=Q_Kernel_12.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=Q_Kernel_12.evaluate)
            else:
                model = QSVC(quantum_kernel=Q_Kernel_12)
                
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_12: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_12.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")

    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_12'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_zz_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import ZZFeatureMap
    qfm_zz = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="full")
    
    print(qfm_zz)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_zz = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_zz)
    
    model = PegasosQSVC(quantum_kernel=Q_Kernel_zz, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for q_kernel_zz_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_zz_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_zz_pegassos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_default_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # What additional backends we have available.
    #for backend in provider.backends():
    #    print(backend)
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_default = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full')
    print(qfm_default)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_default = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_default)
        
    model = PegasosQSVC(quantum_kernel=Q_Kernel_default, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_default_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_default_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_default_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))

    return metrics_dataframe
        

def data_map_8(x: np.ndarray) -> float:
    """
    Define a function map from R^n to R.

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: np.pi*(m * n), x)
    return coeff

def data_map_9(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: (np.pi/2)*(m * n), 1 - x)
    return coeff

def data_map_10(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: np.pi*np.exp(((n - m)*(n - m))/8), x)
    return coeff

def data_map_11(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: (np.pi/3)*(m * n), 1/(np.cos(x)))
    return coeff

def data_map_12(x: np.ndarray) -> float:
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: np.pi*(m * n), np.cos(x))
    return coeff

def q_kernel_8_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):
   
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_8 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_8)
    print(qfm_8)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_8 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_8)
    
    model = PegasosQSVC(quantum_kernel=Q_Kernel_8, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_8_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_8_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_8_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))

    return metrics_dataframe
        

def q_kernel_9_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_9 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_9)
    print(qfm_9)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_9 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_9)
    
    model = PegasosQSVC(quantum_kernel=Q_Kernel_9, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_9_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_9_pegasos.model")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_9_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))

    return metrics_dataframe
        

def q_kernel_10_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_10 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_10)
    print(qfm_10)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_10 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_10)
    
    model = PegasosQSVC(quantum_kernel=Q_Kernel_10, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_10_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_10_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_10_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))

    return metrics_dataframe


def q_kernel_11_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_11 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_11)
    print(qfm_11)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_11 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_11)
    
    model = PegasosQSVC(quantum_kernel=Q_Kernel_11, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_11_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_11_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_11_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))

    return metrics_dataframe

def q_kernel_12_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    from qiskit.circuit.library import PauliFeatureMap
    qfm_12 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_12)
    print(qfm_12)
    
    # For quantum access, the following lines must be adapted
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    fidelity = ComputeUncompute(sampler=sampler)
    Q_Kernel_12 = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm_12)

    model = PegasosQSVC(quantum_kernel=Q_Kernel_12, C=C, num_steps=num_steps)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_12_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_12_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        model.fit(X_train_, y_train_)

        # score classifier
        score[i] = model.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    print("cross validation mean: ", cross_mean)
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_mean, cross_std]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_12_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))

    return metrics_dataframe

"""
Part 2: Quantum Neural Networks

Algorithms from qiskit (https://quantum-computing.ibm.com):
- Variational Quantum Classifier (VQC)
- Classification with an EstimatorQNN
- Classification with a SamplerQNN

Inputs:
- X, y: non-splitted dataset (used for cross-validation)
- X_train, y_train: dataset for model training
- X_test, y_test: dataset for model testing
- cv: number of k-folds for cross-validation
- feature_dimension: dimensionality of the data (equal to the number of required qubits)
- reps: number of times the feature map circuit is repeated
- ibm_account: IBM account (API)
- output_folder: path to save outputs
- quantum_backend: backend options for premium access
    - ibmq_qasm_simulator
    - ibmq_armonk
    - ibmq_santiago
    - ibmq_bogota
    - ibmq_lima
    - ibmq_belem
    - ibmq_quito
    - simulator_statevector
    - simulator_mps
    - simulator_extended_stabilizer
    - simulator_stabilizer
    - ibmq_manila

Output:
- DataFrame with the following metrics:
    - accuracy_score: ratio of correct predictions to total predictions
    - precision_score: number of correct positive predictions
    - recall_score: number of actual positive cases predicted correctly
    - f1_score: harmonic mean of precision and recall
    - cross_val_score: cross-validation score
"""


# Classification with Variational Quantum Classifier (VQC)

"""

VQC is a variant of the NeuralNetworkClassifier with a SamplerQNN which applies a parity mapping or extensions.
By default, VQC will apply CrossEntropyLoww function. The labels are given in one-hot encoded format. We use to_categorical from tensorflow.keras.utils. This will return predictions also in one-hot encoded format. Two important elements of VQC is the feature map and ansatz. Here, we will use the ZZFeatureMap which is a standard feature maps in the Qiskit circuit library.

"""

def q_vqc(X, X_train, X_test, y, y_train, y_test,  number_classes = None, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, cv = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
            
    # One-hot encode target column using tp_categorical
    # A column will be created for each output category.
    #from tensorflow.keras.utils import to_categorical
    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    
    #number_inputs = feature_dimension # Number of qubits
    #output_shape = number_classes  # Number of classes of the dataset
        
    # Callback function that draws a live plot when the .fit() method is called
    #from matplotlib import pyplot as plt
    
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        if output_folder is not None:
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            plt.savefig(output_folder+'Objective_function_value_against_iteration.png')
        else:
            print("No output folder to save figure plotting objective function value against iteration")
            
    # For quantum access, the following lines must be adapted

    # Create feature map, ansatz, and optimizer
    from qiskit.circuit.library import ZZFeatureMap
    feature_map = ZZFeatureMap(feature_dimension)
    from qiskit.circuit.library import RealAmplitudes
    ansatz = RealAmplitudes(feature_dimension, reps=reps)
    
    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1024
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(),
        sampler= sampler,
        callback=callback_graph,
    )
    
    # Create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    
    # fit classifier to data and calculate elpased time
    import time
    start = time.time()
    vqc.fit(X_train, y_train)
    elapsed = time.time() - start

    # score classifier
    vqc.score(X_train, y_train)

    # Predict data points from X_test
    y_predict = vqc.predict(X_test)
    
    if output_folder is not None:
        vqc.save(output_folder+"vqc.model")

    from sklearn import metrics
    from sklearn.metrics import classification_report

    # Print predicted values and real values of the X_test dataset
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_predict)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")

    # Reverse the one hot encoding
    y_test = [np.argmax(y, axis=None, out=None) for y in y_test]
    y_predict = [np.argmax(y, axis=None, out=None) for y in y_predict]
    
    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
 
        # fit classifier to data
        vqc.fit(X_train_, y_train_)

        # score classifier
        score[i] = vqc.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    
    # Print accuracy metrics of the model
    results = [metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predict),metrics.precision_score(y_test, y_predict, average='micro'),metrics.recall_score(y_test, y_predict, average='micro'),metrics.f1_score(y_test, y_predict, average='micro'), cross_mean, cross_std]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_vqc'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_predict))
            
    return metrics_dataframe

"""

Here we will use EstimatorQNN for binary classification within a NeuralNetworkClassifier. The EstimatorQNN will return one-dimensional output in [−1,+1] from two classes that we assigned as {−1,+1}.
 
"""

def q_estimatorqnn(X, X_train, X_test, y, y_train, y_test,  number_classes = None, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, cv = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed
    
    #number_inputs = feature_dimension # Number of qubits
    #output_shape = number_classes  # corresponds to the number of classes
    
    # callback function that draws a live plot when the .fit() method is called
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        if output_folder is not None:
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            plt.savefig(output_folder+'Objective_function_value_against_iteration.png')
        else:
            print("No output folder available to save plot for objective function value against iteration")


    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1024
        options.optimization_level = 3
        estimator = Estimator(session=backend, options = options)
    else:
        from qiskit.primitives import Estimator
        estimator = Estimator()

    # construct feature map
    from qiskit.circuit.library import ZZFeatureMap
    feature_map = ZZFeatureMap(feature_dimension)
    # construct ansatz
    from qiskit.circuit.library import RealAmplitudes
    ansatz = RealAmplitudes(feature_dimension, reps=reps)

    # construct quantum circuit
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(feature_dimension)
    qc.append(feature_map, range(feature_dimension))
    qc.append(ansatz, range(feature_dimension))
    qc.decompose().draw()
        
    # Build QNN
    estimator_qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator = estimator
    )

    # QNN maps inputs to [-1, +1]
    #estimator_qnn.forward(X_train[0, :], algorithm_globals.random.random(estimator_qnn.num_weights))
    
    # Create classifier
    estimator_classifier = NeuralNetworkClassifier(neural_network=estimator_qnn, optimizer=COBYLA(), callback=callback_graph)

    # create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    #plt.rcParams["figure.figsize"] = (12, 6)

    # fit classifier to data
    estimator_classifier.fit(X_train, y_train)

    # return to default figsize
    #plt.rcParams["figure.figsize"] = (6, 4)

    # score classifier
    estimator_classifier.score(X_train, y_train)

    # Predict data points from X_test
    y_predict = estimator_classifier.predict(X_test)

    if output_folder is not None:
        estimator_classifier.save(output_folder+"circuit_classifier.model")

    from sklearn import metrics
    from sklearn.metrics import classification_report

    # Print predicted values and real values of the X_test dataset
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_predict)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")

    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
        
        # fit classifier to data
        estimator_classifier.fit(X_train_, y_train_)

        # score classifier
        score[i] = estimator_classifier.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    
    # Print accuracy metrics of the model
    results = [metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predict),metrics.precision_score(y_test, y_predict, average='micro'),metrics.recall_score(y_test, y_predict, average='micro'),metrics.f1_score(y_test, y_predict, average='micro'), cross_mean, cross_std]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_estimatorqnn'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_predict))
    print(estimator_classifier.weights)
            
    return metrics_dataframe

"""

Here we will use SamplerQNN for  classification within a NeuralNetworkClassifier. The SamplerQNN will return a d-dimensional probability vector where d is the number of classes.
For binary classification we use the parity mapping.

The parity mapping can be the following:

# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2
    
And we can replace the following lines:

        # Build QNN
        sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
        )
        
by

        # Build QNN
        sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=output_shape,
        )
  
"""
    
def q_samplerqnn(X, X_train, X_test, y, y_train, y_test, number_classes = None, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, cv = None, output_folder = None):

    #X_train = X_train.to_numpy() # Convert pandas DataFrame into numpy array
    #y_train = pd.DataFrame(data = y_train, columns = ['Target'])
    #y_train = y_train['Target'].replace(0, -1).to_numpy() # We replace our labels by 1 and -1 and convert pandas DataFrame into numpy array

    #X_test = X_test.to_numpy() # Convert pandas DataFrame into numpy array
    #y_test = pd.DataFrame(data = y_test, columns = ['Target'])
    #y_test = y_test['Target'].replace(0, -1).to_numpy() # We replace our labels by 1 and -1 and convert pandas DataFrame into numpy array
     
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
     
     
    # seed for randomization, to keep outputs consistent
    #seed = 123456
    #algorithm_globals.random_seed = seed
    
    number_inputs = feature_dimension
    
    # callback function that draws a live plot when the .fit() method is called
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        if output_folder is not None:
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            plt.savefig(output_folder+'Objective_function_value_against_iteration.png')
        else:
            print("No output folder available to save plot for objective function value against iteration")


    if quantum_backend is not None:
        # Import QiskitRuntimeService and Sampler
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        # Define service
        service = QiskitRuntimeService(channel = 'ibm_quantum', token = ibm_account, instance = 'ibm-q/open/main')
        # Get backend
        backend = service.backend(quantum_backend) # Use a simulator or hardware from the cloud
        # Define Sampler: With our training and testing datasets ready, we set up the FidelityQuantumKernel class to calculate a kernel matrix using the ZZFeatureMap. We use the reference implementation of the Sampler primitive and the ComputeUncompute fidelity that computes overlaps between states. These are the default values and if you don't pass a Sampler or Fidelity instance, the same objects will be created automatically for you.
        # Run Quasi-Probability calculation
        # optimization_level=3 adds dynamical decoupling
        # resilience_level=1 adds readout error mitigation
        from qiskit_ibm_runtime import Options
        options = Options()
        options.resilience_level = 1
        options.execution.shots = 1024
        options.optimization_level = 3
        sampler = Sampler(session=backend, options = options)
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
    
    # construct feature map
    from qiskit.circuit.library import ZZFeatureMap
    feature_map = ZZFeatureMap(feature_dimension)

    # construct ansatz
    from qiskit.circuit.library import RealAmplitudes
    ansatz = RealAmplitudes(feature_dimension, reps=reps)
    
    # construct quantum circuit
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(feature_dimension)
    qc.append(feature_map, range(feature_dimension))
    qc.append(ansatz, range(feature_dimension))
    qc.decompose().draw()
            
    # Build QNN
    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        output_shape=number_classes,
        sampler=sampler
    )

    # construct classifier
    sampler_classifier = NeuralNetworkClassifier(
    neural_network=sampler_qnn, optimizer=COBYLA(maxiter=30), callback=callback_graph
    )

    # create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    
    # fit classifier to data
    sampler_classifier.fit(X_train, y_train)

    # score classifier
    sampler_classifier.score(X_train, y_train)

    # evaluate data points
    y_predict = sampler_classifier.predict(X_test)
    
    if output_folder is not None:
        sampler_classifier.save(output_folder+"opflow_classifier.model")

    from sklearn import metrics
    from sklearn.metrics import classification_report

    # Print predicted values and real values of the X_test dataset
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_predict)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")

    # K-Fold Cross Validation
    from sklearn.model_selection import KFold
    k_fold = KFold(n_splits=cv)
    score = np.zeros(cv)
    i = 0
    print(score)
    for indices_train, indices_test in k_fold.split(X_train):
        #print(indices_train, indices_test)
        X_train_ = X_train[indices_train]
        X_test_ = X_train[indices_test]
        y_train_ = y_train[indices_train]
        y_test_ = y_train[indices_test]
        
        # fit classifier to data
        sampler_classifier.fit(X_train_, y_train_)

        # score classifier
        score[i] = sampler_classifier.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    
    # Print accuracy metrics of the model
    results = [metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predict),metrics.precision_score(y_test, y_predict, average='micro'),metrics.recall_score(y_test, y_predict, average='micro'),metrics.f1_score(y_test, y_predict, average='micro'), cross_mean, cross_std]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_samplerqnn'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_predict))
    print("sampler_classifier.weights")
            
    return metrics_dataframe


