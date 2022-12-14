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

# Loading your IBM Quantum account(s)
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# Import utilities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce

# sklearn imports
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Importing standard Qiskit libraries and Qiskit Machine Learning imports
from qiskit import Aer, QuantumCircuit, BasicAer
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import circuit_drawer
from qiskit.algorithms.optimizers import SPSA

from qiskit.opflow import Z, I, StateFn
from qiskit.circuit import Parameter

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC

from typing import Union
from qiskit_machine_learning.exceptions import QiskitMachineLearningError



"""
Quantum Algotihms for classification.

We use the encoding function from Havli??c??ek et al. (q_kernel_zz) and five encoding functions presented by Suzuki et al. (q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_default) and apply SVC from scikit-learn.

We also use the encoding function from Havli??c??ek et al. (q_kernel_default_pegasos) and five encoding functions presented by Suzuki et al. (q_kernel_8_pegasos, q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos) and apply Pegasos algorithm from Shalev-Shwartz.

We can also set up the QuantumKernel class to calculate a kernel matrix using the ZZFeatureMap (q_kernel_zz) with SVC from scikit-learn or pegasos algorithm (q_kernel_zz_pegasos)

Algorithms: q_kernel_default, q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_default_pegasos, q_kernel_8_pegasos, q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos, q_kernel_zz_pegasos, q_kernel_training


Inputs:
    X,y non splitted dataset separated by features (X) and labels (y). This is used for cross-validation.
    X_train, y_train selected dataset to train the model separated by features (X_train) and labels (y_train)
    X_test, y_test: selected dataset to test the model speparated by features (X_test) and labels (y_test)
    cv: number of k-folds for cross-validation
    feature_dimension = dimensionality of the data which equal to the number of required qubits,
    reps= number of times the feature map circuit is repeated,
    ibm_account = None,
    quantum_backend (different if premium access):
        ibmq_qasm_simulator
        ibmq_armonk
        ibmq_santiago
        ibmq_bogota
        ibmq_lima
        ibmq_belem
        ibmq_quito
        simulator_statevector
        simulator_mps
        simulator_extended_stabilizer
        simulator_stabilizer
        ibmq_manila
        
    For pegasos algorithms:
        tau = number of steps performed during the training procedure
        C = regularization parameter

    Output:
        A DataFrame with the following metrics:
            accuracy_score: It is calculated as the ratio of the number of correct predictions to all number predictions made by the classifiers.
            precision_score: Precision is the number of correct outputs or how many of the correctly predicted cases turned out to be positive.
            recall_score: Recall defines how many of the actual positive cases we were able to predict correctly.
            f1_score: It is a harmonic mean of precision and recall.
            cross_val_score: Cross-validation score

"""


def q_kernel_zz(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):
   
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_zz = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="full")
    #qfm_zz = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="linear")
    #adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
            
    print(qfm_zz)
    
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        sim = provider.backends.ibmq_qasm_simulator
        qcomp_backend = QuantumInstance(sim, shots=8192, seed_simulator=seed, seed_transpiler=seed)
        #qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_zz = QuantumKernel(feature_map=qfm_zz, quantum_instance=qcomp_backend)
                
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_zz = QuantumKernel(feature_map=qfm_zz, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Use of least busy quantum hardwarr
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_zz = QuantumKernel(feature_map=qfm_zz, quantum_instance=real_qcomp_backend)

            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_zz = QuantumKernel(feature_map=qfm_zz, quantum_instance=real_qcomp_backend)
    
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
    
    
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for q_kernel_zz: {score}')
          
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"q_kernel_zz.model")
    
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
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_zz'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe


def q_kernel_default(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_default = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full')
    print(qfm_default)
        
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of a simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_default = QuantumKernel(feature_map=qfm_default, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_default = QuantumKernel(feature_map=qfm_default, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Backend objects can also be set up using the IBMQ package.
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_default = QuantumKernel(feature_map=qfm_default, quantum_instance=real_qcomp_backend)

            else:
                # Use of a real quantum computer
                # Backend objects can also be set up using the IBMQ package.
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_default = QuantumKernel(feature_map=qfm_default, quantum_instance=real_qcomp_backend)


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
    seed = 123456
    algorithm_globals.random_seed = seed



    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_8 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_8)
    print(qfm_8)

    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_8 = QuantumKernel(feature_map=qfm_8, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            
            # Backend objects can also be set up using the IBMQ package.
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_8 = QuantumKernel(feature_map=qfm_8, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_8 = QuantumKernel(feature_map=qfm_8, quantum_instance=real_qcomp_backend)

            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_8 = QuantumKernel(feature_map=qfm_8, quantum_instance=real_qcomp_backend)
            

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
    seed = 123456
    algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_9 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_9)
    print(qfm_9)

    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_9 = QuantumKernel(feature_map=qfm_9, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_9 = QuantumKernel(feature_map=qfm_9, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Use of the least busy quantum hardware
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_9 = QuantumKernel(feature_map=qfm_9, quantum_instance=real_qcomp_backend)

            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_9 = QuantumKernel(feature_map=qfm_9, quantum_instance=real_qcomp_backend)

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
    seed = 123456
    algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_10 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_10)
    print(qfm_10)
    
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_10 = QuantumKernel(feature_map=qfm_10, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # Backend objects can also be set up using the IBMQ package.
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_10 = QuantumKernel(feature_map=qfm_10, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Use of the least busy quantum hardware
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_10 = QuantumKernel(feature_map=qfm_10, quantum_instance=real_qcomp_backend)
            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_10 = QuantumKernel(feature_map=qfm_10, quantum_instance=real_qcomp_backend)
        
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
    seed = 123456
    algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_11 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_11)
    print(qfm_11)


    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_11 = QuantumKernel(feature_map=qfm_11, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_11 = QuantumKernel(feature_map=qfm_11, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Backend objects can also be set up using the IBMQ package.
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_11 = QuantumKernel(feature_map=qfm_11, quantum_instance=real_qcomp_backend)
            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_11 = QuantumKernel(feature_map=qfm_11, quantum_instance=real_qcomp_backend)

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
    qfm_12 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_12)
    print(qfm_12)
    
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_12 = QuantumKernel(feature_map=qfm_12, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
        
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            Q_Kernel_12 = QuantumKernel(feature_map=qfm_12, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Use of the least busy quantum hardware
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_12 = QuantumKernel(feature_map=qfm_12, quantum_instance=real_qcomp_backend)
            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                Q_Kernel_12 = QuantumKernel(feature_map=qfm_12, quantum_instance=real_qcomp_backend)
                
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


    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_zz = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="full")
    
    print(qfm_zz)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_zz = QuantumKernel(feature_map=qfm_zz, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_zz = QuantumKernel(feature_map=qfm_zz, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_zz.evaluate, C=C, num_steps=tau)
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
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_zz_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe



def q_kernel_default_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_default = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full')
    print(qfm_default)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_default = QuantumKernel(feature_map=qfm_default, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_default = QuantumKernel(feature_map=qfm_default, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_default.evaluate, C=C, num_steps=tau)
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
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
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

    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed



    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_8 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_8)
    print(qfm_8)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_8 = QuantumKernel(feature_map=qfm_8, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_8 = QuantumKernel(feature_map=qfm_8, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_8.evaluate, C=C, num_steps=tau)
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
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_8_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_9_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_9 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_9)
    print(qfm_9)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_9 = QuantumKernel(feature_map=qfm_9, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_9 = QuantumKernel(feature_map=qfm_9, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_9.evaluate, C=C, num_steps=tau)
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    print(f'Callable kernel classification test score for Q_Kernel_9_pegasos: {score}')
        
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        model.save(output_folder+"q_kernel_9_pegasos.model")
    
    print("\n")
    print("Print predicted data coming from X_test as new input data")
    print(y_pred)
    print("\n")
    print("Print real values\n")
    print(y_test)
    print("\n")
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_9_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_10_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_10 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_10)
    print(qfm_10)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_10 = QuantumKernel(feature_map=qfm_10, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_10 = QuantumKernel(feature_map=qfm_10, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_10.evaluate, C=C, num_steps=tau)
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
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_10_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe


def q_kernel_11_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_11 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_11)
    print(qfm_11)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_11 = QuantumKernel(feature_map=qfm_11, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_11 = QuantumKernel(feature_map=qfm_11, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_11.evaluate, C=C, num_steps=tau)
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
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_11_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe

def q_kernel_12_pegasos(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, C= None, num_steps= None, output_folder = None):

    # Backend objects can also be set up using the IBMQ package.
    # The use of these requires us to sign with an IBMQ account.
    # Assuming the credentials are already loaded onto your computer, you sign in with
    IBMQ.save_account(ibm_account, overwrite=True)

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
    #provider = IBMQ.get_provider(hub='ibm-q')
    #backend = provider.get_backend(quantum_backend)

    # What additional backends we have available.
    for backend in provider.backends():
        print(backend)
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed


    # Quantum Feature Mapping with feature_dimension = 2 and reps = 2
    qfm_12 = PauliFeatureMap(feature_dimension=feature_dimension,
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=reps, entanglement='full', data_map_func=data_map_12)
    print(qfm_12)
    
    if 'simulator_statevector' in quantum_backend:
        # Use of a simulator
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        Q_Kernel_12 = QuantumKernel(feature_map=qfm_12, quantum_instance=qcomp_backend)
    else:
        # Use of a real quantum computer
        real_qcomp_backend = QuantumInstance(provider.get_backend(quantum_backend), shots=1024)
        Q_Kernel_12 = QuantumKernel(feature_map=qfm_12, quantum_instance=real_qcomp_backend)

    model = PegasosQSVC(kernel=Q_Kernel_12.evaluate, C=C, num_steps=tau)
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
    
    results = [metrics.accuracy_score(y_test, y_pred),metrics.precision_score(y_test, y_pred, average='micro'),metrics.recall_score(y_test, y_pred, average='micro'),metrics.f1_score(y_test, y_pred, average='micro'), cross_val_score(model, X_train, y_train, cv=cv).mean(), cross_val_score(model, X_train, y_train, cv=cv).std()]
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_12_pegasos'])

    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
        
    return metrics_dataframe
    


def q_twolayerqnn(X, X_train, X_test, y, y_train, y_test, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, cv = None, output_folder = None):

    X_train = X_train.to_numpy() # Convert pandas DataFrame into numpy array
    y_train = pd.DataFrame(data = y_train, columns = ['Target'])
    y_train = y_train['Target'].replace(0, -1).to_numpy() # We replace our labels by 1 and -1 and convert pandas DataFrame into numpy array

    X_test = X_test.to_numpy() # Convert pandas DataFrame into numpy array
    y_test = pd.DataFrame(data = y_test, columns = ['Target'])
    y_test = y_test['Target'].replace(0, -1).to_numpy() # We replace our labels by 1 and -1 and convert pandas DataFrame into numpy array
            
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed
    
    number_inputs = feature_dimension
        
    # callback function that draws a live plot when the .fit() method is called
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
    
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of a simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
            
        # Define a quantum instance
        quantum_instance = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        
        # Build our QNN
        opflow_qnn = TwoLayerQNN(number_inputs, quantum_instance=quantum_instance)

        # QNN maps inputs to [-1, +1]
        opflow_qnn.forward(X_train[0, :], algorithm_globals.random.random(opflow_qnn.num_weights))

    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            
            # Backend objects can also be set up using the IBMQ package.
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)

            # Define a quantum instance
            quantum_instance = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
                        
            # Build our QNN
            opflow_qnn = TwoLayerQNN(number_inputs, quantum_instance=quantum_instance)

            # QNN maps inputs to [-1, +1]
            opflow_qnn.forward(X_train[0, :], algorithm_globals.random.random(opflow_qnn.num_weights))
        
        else:
            if 'least_busy' in quantum_backend:
                # Use the least busy quantum hardware
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than feature_dimension qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                
                # Define a quantum instance
                quantum_instance = QuantumInstance(backend, shots=1024)
                        
                # Build our QNN
                opflow_qnn = TwoLayerQNN(number_inputs, quantum_instance=quantum_instance)

                # QNN maps inputs to [-1, +1]
                opflow_qnn.forward(X_train[0, :], algorithm_globals.random.random(opflow_qnn.num_weights))

            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                
                # Define a quantum instance
                quantum_instance = QuantumInstance(backend, shots=1024)
                        
                # Build our QNN
                opflow_qnn = TwoLayerQNN(number_inputs, quantum_instance=quantum_instance)

                # QNN maps inputs to [-1, +1]
                opflow_qnn.forward(X_train[0, :], algorithm_globals.random.random(opflow_qnn.num_weights))
    
    # Create the neural network classifier
    opflow_classifier = NeuralNetworkClassifier(opflow_qnn, optimizer=COBYLA(), callback=callback_graph)

    # create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    
    # fit classifier to data
    opflow_classifier.fit(X_train, y_train)

    # score classifier
    opflow_classifier.score(X_test, y_test)

    # evaluate data points
    y_predict = opflow_classifier.predict(X_test)
    
    if output_folder is not None:
        opflow_classifier.save(output_folder+"opflow_classifier.model")

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
        opflow_classifier.fit(X_train_, y_train_)

        # score classifier
        score[i] = opflow_classifier.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    
    # Print accuracy metrics of the model
    results = [metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predict),metrics.precision_score(y_test, y_predict, average='micro'),metrics.recall_score(y_test, y_predict, average='micro'),metrics.f1_score(y_test, y_predict, average='micro'), cross_mean, cross_std]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_twolayerqnn'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_predict))
            
    return metrics_dataframe


def q_circuitqnn(X, X_train, X_test, y, y_train, y_test,  number_classes = None, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, cv = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed
    
    number_inputs = feature_dimension # Number of qubits
    output_shape = number_classes  # corresponds to the number of classes
    
    # callback function that draws a live plot when the .fit() method is called
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
            
        # Define a quantum instance
        quantum_instance = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        
        # construct feature map
        feature_map = ZZFeatureMap(number_inputs)

        # construct ansatz
        ansatz = RealAmplitudes(number_inputs, reps=reps)

        # construct quantum circuit
        qc = QuantumCircuit(number_inputs)
        qc.append(feature_map, range(number_inputs))
        qc.append(ansatz, range(number_inputs))
        qc.decompose().draw()
            
        # Build QNN
        circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            output_shape=output_shape,
            quantum_instance=quantum_instance,
        )

    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)

            # Define a quantum instance
            quantum_instance = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
                        
            # construct feature map
            feature_map = ZZFeatureMap(number_inputs)

            # construct ansatz
            ansatz = RealAmplitudes(number_inputs, reps=reps)

            # construct quantum circuit
            qc = QuantumCircuit(number_inputs)
            qc.append(feature_map, range(number_inputs))
            qc.append(ansatz, range(number_inputs))
            qc.decompose().draw()
            
            # Build QNN
            circuit_qnn = CircuitQNN(
                circuit=qc,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
                output_shape=output_shape,
                quantum_instance=quantum_instance,
            )
        
        else:
            if 'least_busy' in quantum_backend:
                # Use of the least busy real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than feature_dimension qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                
                # Define a quantum instance
                quantum_instance = QuantumInstance(backend, shots=1024)
                        
                # construct feature map
                feature_map = ZZFeatureMap(number_inputs)

                # construct ansatz
                ansatz = RealAmplitudes(number_inputs, reps=reps)

                # construct quantum circuit
                qc = QuantumCircuit(number_inputs)
                qc.append(feature_map, range(number_inputs))
                qc.append(ansatz, range(number_inputs))
                qc.decompose().draw()
                
                # Build QNN
                circuit_qnn = CircuitQNN(
                    circuit=qc,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
            
                    output_shape=output_shape,
                    quantum_instance=quantum_instance,
                )

            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                
                # Define a quantum instance
                quantum_instance = QuantumInstance(backend, shots=1024)

                # construct feature map
                feature_map = ZZFeatureMap(number_inputs)

                # construct ansatz
                ansatz = RealAmplitudes(number_inputs, reps=reps)

                # construct quantum circuit
                qc = QuantumCircuit(number_inputs)
                qc.append(feature_map, range(number_inputs))
                qc.append(ansatz, range(number_inputs))
                qc.decompose().draw()
                
                # Build QNN
                circuit_qnn = CircuitQNN(
                    circuit=qc,
                    input_params=feature_map.parameters,
                    weight_params=ansatz.parameters,
                    output_shape=output_shape,
                    quantum_instance=quantum_instance,
                )

    # Create classifier
    circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn, optimizer=COBYLA(), callback=callback_graph)

    # create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    #plt.rcParams["figure.figsize"] = (12, 6)

    # fit classifier to data
    circuit_classifier.fit(X_train, y_train)

    # return to default figsize
    #plt.rcParams["figure.figsize"] = (6, 4)

    # score classifier
    circuit_classifier.score(X_test, y_test)

    # Predict data points from X_test
    y_predict = circuit_classifier.predict(X_test)

    if output_folder is not None:
        circuit_classifier.save(output_folder+"circuit_classifier.model")

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
        circuit_classifier.fit(X_train_, y_train_)

        # score classifier
        score[i] = circuit_classifier.score(X_test_, y_test_)
        i = i + 1

    import math
    print("cross validation scores: ", score)
    cross_mean = sum(score) / len(score)
    cross_var = sum(pow(x - cross_mean,2) for x in score) / len(score)  # variance
    cross_std  = math.sqrt(cross_var)  # standard deviation
    
    # Print accuracy metrics of the model
    results = [metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_predict),metrics.precision_score(y_test, y_predict, average='micro'),metrics.recall_score(y_test, y_predict, average='micro'),metrics.f1_score(y_test, y_predict, average='micro'), cross_mean, cross_std]
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_circuitqnn'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_predict))
            
    return metrics_dataframe
    
# Classification with Variational Quantum Classifier (VQC)
def q_vqc(X, X_train, X_test, y, y_train, y_test,  number_classes = None, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, cv = None, output_folder = None):

    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
            
    # One-hot encode target column using tp_categorical
    # A column will be created for each output category.
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
                    
    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed
    
    number_inputs = feature_dimension # Number of qubits
    output_shape = number_classes  # Number of classes of the dataset
    
    # Callback function that draws a live plot when the .fit() method is called
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        
    if 'ibmq_qasm_simulator' in quantum_backend:
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
            
        # Define a quantum instance
        quantum_instance = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        
        # Create VQC
        # Create feature map, ansatz, and optimizer
        feature_map = ZZFeatureMap(number_inputs)
        ansatz = RealAmplitudes(number_inputs, reps=reps)

        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            loss="cross_entropy",
            optimizer=COBYLA(),
            quantum_instance=quantum_instance,
            callback=callback_graph,
        )

    else:
        if 'statevector_simulator' in quantum_backend:
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)

            # Define a quantum instance
            quantum_instance = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
                    
            # Create VQC
            # Create feature map, ansatz, and optimizer
            feature_map = ZZFeatureMap(number_inputs)
            ansatz = RealAmplitudes(number_inputs, reps=reps)

            vqc = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                loss="cross_entropy",
                optimizer=COBYLA(),
                quantum_instance=quantum_instance,
                callback=callback_graph,
            )
        
        else:
            # Use of the least busy real quantum computer
            if 'least_busy' in quantum_backend:
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than feature_dimension qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                
                # Define a quantum instance
                quantum_instance = QuantumInstance(backend, shots=1024)
                        
                # Create feature map, ansatz, and optimizer
                feature_map = ZZFeatureMap(number_inputs)
                ansatz = RealAmplitudes(number_inputs, reps=reps)

                # Create VQC
                vqc = VQC(
                    feature_map=feature_map,
                    ansatz=ansatz,
                    loss="cross_entropy",
                    optimizer=COBYLA(),
                    quantum_instance=quantum_instance,
                    callback=callback_graph,
                )
            
            else:
                # Use of a selected real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                
                # Define a quantum instance
                quantum_instance = QuantumInstance(backend, shots=1024)

                # Create VQC
                # Create feature map, ansatz, and optimizer
                feature_map = ZZFeatureMap(number_inputs)
                ansatz = RealAmplitudes(number_inputs, reps=reps)

                vqc = VQC(
                    feature_map=feature_map,
                    ansatz=ansatz,
                    loss="cross_entropy",
                    optimizer=COBYLA(),
                    quantum_instance=quantum_instance,
                    callback=callback_graph,
                )
                
    # Create empty array for callback to store evaluations of the objective function
    objective_func_vals = []
    
    # fit classifier to data
    vqc.fit(X_train, y_train)

    # score classifier
    vqc.score(X_test, y_test)

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



def q_kernel_training_orginal(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):
 
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Define a callback class for our optimizer
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

    # normalize the data between 0 and 2pi
    X_train -= X_train.min(0)
    X_train /= X_train.max(0)
    X_train *= 2*np.pi

    # normalize the data between 0 and 2pi
    X_test -= X_test.min(0)
    X_test /= X_test.max(0)
    X_test *= 2*np.pi
    
    if feature_dimension == 2:
        # Create a rotational layer to train. We will rotate each qubit the same amount.
        user_params = ParameterVector("??", feature_dimension)
        print("feature dimension : %i"%feature_dimension)
        fm0 = QuantumCircuit(feature_dimension)
        fm0.ry(user_params[0], 0)
        fm0.ry(user_params[1], 1)
    else:
        if feature_dimension == 5:
            # Create a rotational layer to train. We will rotate each qubit the same amount.
            user_params = ParameterVector("??", feature_dimension)
            fm0 = QuantumCircuit(feature_dimension)
            fm0.ry(user_params[0], 0)
            fm0.ry(user_params[1], 1)
            fm0.ry(user_params[2], 2)
            fm0.ry(user_params[3], 3)
            fm0.ry(user_params[4], 4)
        else:
            if feature_dimension == 6:
                # Create a rotational layer to train. We will rotate each qubit the same amount.
                user_params = ParameterVector("??", feature_dimension)
                fm0 = QuantumCircuit(feature_dimension)
                fm0.ry(user_params[0], 0)
                fm0.ry(user_params[1], 1)
                fm0.ry(user_params[2], 2)
                fm0.ry(user_params[3], 3)
                fm0.ry(user_params[4], 4)
                fm0.ry(user_params[5], 5)
            else:
                if feature_dimension == 10:
                    # Create a rotational layer to train. We will rotate each qubit the same amount.
                    user_params = ParameterVector("??", feature_dimension)
                    fm0 = QuantumCircuit(feature_dimension)
                    fm0.ry(user_params[0], 0)
                    fm0.ry(user_params[1], 1)
                    fm0.ry(user_params[2], 2)
                    fm0.ry(user_params[3], 3)
                    fm0.ry(user_params[4], 4)
                    fm0.ry(user_params[5], 5)
                    fm0.ry(user_params[6], 6)
                    fm0.ry(user_params[7], 7)
                    fm0.ry(user_params[8], 8)
                    fm0.ry(user_params[9], 9)
                else:
                    if feature_dimension == 20:
                        # Create a rotational layer to train. We will rotate each qubit the same amount.
                        user_params = ParameterVector("??", feature_dimension)
                        fm0 = QuantumCircuit(feature_dimension)
                        fm0.ry(user_params[0], 0)
                        fm0.ry(user_params[1], 1)
                        fm0.ry(user_params[2], 2)
                        fm0.ry(user_params[3], 3)
                        fm0.ry(user_params[4], 4)
                        fm0.ry(user_params[5], 5)
                        fm0.ry(user_params[6], 6)
                        fm0.ry(user_params[7], 7)
                        fm0.ry(user_params[8], 8)
                        fm0.ry(user_params[9], 9)
                        fm0.ry(user_params[10], 10)
                        fm0.ry(user_params[11], 11)
                        fm0.ry(user_params[12], 12)
                        fm0.ry(user_params[13], 13)
                        fm0.ry(user_params[14], 14)
                        fm0.ry(user_params[15], 15)
                        fm0.ry(user_params[16], 16)
                        fm0.ry(user_params[17], 17)
                        fm0.ry(user_params[18], 18)
                        fm0.ry(user_params[19], 19)
                    else:
                        if feature_dimension == 32:
                            # Create a rotational layer to train. We will rotate each qubit the same amount.
                            user_params = ParameterVector("??", feature_dimension)
                            fm0 = QuantumCircuit(feature_dimension)
                            fm0.ry(user_params[0], 0)
                            fm0.ry(user_params[1], 1)
                            fm0.ry(user_params[2], 2)
                            fm0.ry(user_params[3], 3)
                            fm0.ry(user_params[4], 4)
                            fm0.ry(user_params[5], 5)
                            fm0.ry(user_params[6], 6)
                            fm0.ry(user_params[7], 7)
                            fm0.ry(user_params[8], 8)
                            fm0.ry(user_params[9], 9)
                            fm0.ry(user_params[10], 10)
                            fm0.ry(user_params[11], 11)
                            fm0.ry(user_params[12], 12)
                            fm0.ry(user_params[13], 13)
                            fm0.ry(user_params[14], 14)
                            fm0.ry(user_params[15], 15)
                            fm0.ry(user_params[16], 16)
                            fm0.ry(user_params[17], 17)
                            fm0.ry(user_params[18], 18)
                            fm0.ry(user_params[19], 19)
                            fm0.ry(user_params[20], 20)
                            fm0.ry(user_params[21], 21)
                            fm0.ry(user_params[22], 22)
                            fm0.ry(user_params[23], 23)
                            fm0.ry(user_params[24], 24)
                            fm0.ry(user_params[25], 25)
                            fm0.ry(user_params[26], 26)
                            fm0.ry(user_params[27], 27)
                            fm0.ry(user_params[28], 28)
                            fm0.ry(user_params[29], 29)
                            fm0.ry(user_params[30], 30)
                            fm0.ry(user_params[31], 31)

    # Use ZZFeatureMap to represent input data
    fm1 = ZZFeatureMap(feature_dimension=feature_dimension, reps=2, entanglement="linear")

    # Create the feature map, composed of our two circuits
    fm = fm0.compose(fm1)

    fm.decompose().draw(output="mpl")
    print(f"Trainable parameters: {user_params}")
            
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        
        # Instantiate quantum kernel
        quant_kernel = QuantumKernel(fm, user_parameters=user_params, quantum_instance=qcomp_backend)
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
            
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
            quant_kernel = QuantumKernel(fm, user_parameters=user_params, quantum_instance=qcomp_backend)
        else:
            if 'least_busy' in quantum_backend:
                # Use of the least busy quantum hardware
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                quant_kernel = QuantumKernel(fm, user_parameters=user_params, quantum_instance=real_qcomp_backend)

            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                real_qcomp_backend = QuantumInstance(backend, shots=1024)
                quant_kernel = QuantumKernel(fm, user_parameters=user_params, quantum_instance=real_qcomp_backend)

    print("Set up the optimizer: Start")
    # Set up the optimizer
    cb_qkt = QKTCallback()
    spsa_opt = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)

    print("Instantiate a quantum kernel trainer: Start")
    # Instantiate a quantum kernel trainer.
    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel, loss="svc_loss", optimizer=spsa_opt, initial_point=[np.pi / 2]*feature_dimension
    )

    print("Train the kernel using QKT directly")
    # Train the kernel using QKT directly
    qka_results = qkt.fit(X_train, y_train)
    optimized_kernel = qka_results.quantum_kernel
    print(qka_results)

    if multiclass == 'OneVsRestClassifier':
        from sklearn.multiclass import OneVsRestClassifier
        model = OneVsRestClassifier(SVC(kernel=optimized_kernel.evaluate))
    else:
        if multiclass == 'OneVsOneClassifier':
            from sklearn.multiclass import OneVsOneClassifier
            model = OneVsOneClassifier(SVC(kernel=optimized_kernel.evaluate))
        else:
            if multiclass == 'svc':
                model = SVC(kernel=optimized_kernel.evaluate)
            else:
                print("QSVC selected")
                model = QSVC(quantum_kernel=optimized_kernel)
    

    # Use QSVC for classification
    #model = QSVC(quantum_kernel=optimized_kernel)

    # Fit the QSVC
    model.fit(X_train, y_train)

    # Predict the labels
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"test.model")
        
    # Evalaute the test accuracy
    accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"accuracy test: {accuracy_test}")

    # Print predicted values and real values of the X_test dataset
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
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_training'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
            
    return metrics_dataframe

def q_kernel_training_version_1(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):
 
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # Define a callback class for our optimizer
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

    # normalize the data between 0 and 2pi
    X_train -= X_train.min(0)
    X_train /= X_train.max(0)
    X_train *= 2*np.pi

    # normalize the data between 0 and 2pi
    X_test -= X_test.min(0)
    X_test /= X_test.max(0)
    X_test *= 2*np.pi
    
    
    # Create 2-qubit feature map
    qc = QuantumCircuit(feature_dimension)

    # Vectors of input and trainable user parameters
    input_params = ParameterVector("x_par", feature_dimension)
    user_params = ParameterVector("??_par", feature_dimension)
    
    # Create an initial rotation layer of trainable parameters
    for i, param in enumerate(user_params):
        qc.ry(param, qc.qubits[i])

    # Create a rotation layer of input parameters
    for i, param in enumerate(input_params):
        qc.rz(param, qc.qubits[i])
    
    IBMQ.save_account(ibm_account, overwrite=True)
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    # What additional backends we have available.
    qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
    quant_kernel = QuantumKernel(qc, user_parameters=user_params, quantum_instance=qcomp_backend)
    
    cb_qkt = QKTCallback()
    loss_func = "svc_loss"
    optimizer = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)
    initial_point = [np.pi / 2]*feature_dimension

    qk_trainer = QuantumKernelTrainer(
                                quantum_kernel=quant_kernel,
                                loss=loss_func,
                                optimizer=optimizer,
                                initial_point=initial_point,
                                )
    qkt_results = qk_trainer.fit(X_train, y_train)
    optimized_kernel = qkt_results.quantum_kernel

    print("QSVC selected")
    model = QSVC(quantum_kernel=optimized_kernel)
    
    # Fit the QSVC
    model.fit(X_train, y_train)

    # Predict the labels
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"test.model")
        
    # Evalaute the test accuracy
    accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"accuracy test: {accuracy_test}")

    # Print predicted values and real values of the X_test dataset
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
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_training'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
            
    return metrics_dataframe
    
def q_kernel_training(X, X_train, X_test, y, y_train, y_test, cv, feature_dimension = None, reps= None, ibm_account = None, quantum_backend = None, multiclass = None, output_folder = None):
 
    # We convert pandas DataFrame into numpy array
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # seed for randomization, to keep outputs consistent
    seed = 123456
    algorithm_globals.random_seed = seed

    # normalize the data between 0 and 2pi
    X_train -= X_train.min(0)
    X_train /= X_train.max(0)
    X_train *= 2*np.pi

    # normalize the data between 0 and 2pi
    X_test -= X_test.min(0)
    X_test /= X_test.max(0)
    X_test *= 2*np.pi
    
    # Number of parameters equal to number of features
    num_params = feature_dimension

    # Create vector containing number of trainable parameters
    user_params = ParameterVector('??', num_params)
        
    # Create circuit for num_features qbit system
    fm0 = QuantumCircuit(feature_dimension)

    # First feature map component comprises a Y rotation with parameter theta on both qubit regs
    for qubit in range(feature_dimension):
        fm0.ry(user_params[(qubit%num_params)], qubit)
    
    # Use ZZFeatureMap to represent input data
    fm1 = ZZFeatureMap(feature_dimension=feature_dimension)

    # Compose both maps to create one feature map circuit
    feature_map = fm0.compose(fm1)
    
    if 'ibmq_qasm_simulator' in quantum_backend:
        # Use of simulator
        # The use of these requires us to sign with an IBMQ account.
        # Assuming the credentials are already loaded onto your computer, you sign in with
        IBMQ.save_account(ibm_account, overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        # What additional backends we have available.
        for backend in provider.backends():
            print(backend)
        
        qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)
        
    else:
        if 'statevector_simulator' in quantum_backend:
            # Use of a simulator
            # The use of these requires us to sign with an IBMQ account.
            # Assuming the credentials are already loaded onto your computer, you sign in with
            IBMQ.save_account(ibm_account, overwrite=True)
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q')
            # What additional backends we have available.
            for backend in provider.backends():
                print(backend)
            
            qcomp_backend = QuantumInstance(BasicAer.get_backend(quantum_backend), shots=1024, seed_simulator=seed, seed_transpiler=seed)

        else:
            if 'least_busy' in quantum_backend:
                # Use of the least busy quantum hardware
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                
                device = least_busy(provider.backends(
                        filters=lambda x: x.configuration().n_qubits >= feature_dimension   # More than 5 qubits
                        and not x.configuration().simulator                                 # Not a simulator
                        and x.status().operational == True                                  # Operational backend
                        )
                    )
                                # Use of a real quantum computer
                print("Available device: ", device)
                quantum_backend = "%s"%device
                
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                qcomp_backend = QuantumInstance(backend, shots=1024)


            else:
                # Use of a real quantum computer
                # The use of these requires us to sign with an IBMQ account.
                # Assuming the credentials are already loaded onto your computer, you sign in with
                IBMQ.save_account(ibm_account, overwrite=True)
                IBMQ.load_account()
                provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
                # What additional backends we have available.
                for backend in provider.backends():
                    print(backend)
                    
                backend = provider.get_backend(quantum_backend)
                #backend.configuration().default_rep_delay == 0.00001  # Equality test on float is bad
                qcomp_backend = QuantumInstance(backend, shots=1024)
        
    # Instantiate quantum kernel
    quant_kernel = QuantumKernel(feature_map, user_parameters=user_params, quantum_instance=qcomp_backend)

    # Set up model optimizer
    spsa_opt = SPSA(maxiter=10, learning_rate=0.05, perturbation=0.05)

    from qiskit_machine_learning.utils.loss_functions import SVCLoss
    loss_func = SVCLoss(C=1.0)
    print("Initiate a quantum kernel trainer")
    qkt = QuantumKernelTrainer(quantum_kernel=quant_kernel, loss=loss_func, optimizer=spsa_opt, initial_point=[np.pi/2 for i in range(len(user_params))])
    
    # Instantiate a quantum kernel trainer
    #qkt = QuantumKernelTrainer(
    #quantum_kernel=quant_kernel, loss="svc_loss", optimizer=spsa_opt, initial_point=[np.pi/2 for i in range(len(user_params))])
          
    # Use QuantumKernelTrainer object to fit model to training data
    print("QuantumKernelTrainer object to fit model to training data")
    qka_results = qkt.fit(X_train, y_train)
    optimized_kernel = qka_results.quantum_kernel

    # Use QSVC for classification
    model = QSVC(quantum_kernel=optimized_kernel)

    # Fit the QSVC
    model.fit(X_train, y_train)

    # Predict the labels
    y_pred = model.predict(X_test)
    
    if output_folder is not None:
        if multiclass is None:
            model.save(output_folder+"test.model")
        
    # Evalaute the test accuracy
    accuracy_test = metrics.balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"accuracy test: {accuracy_test}")

    # Print predicted values and real values of the X_test dataset
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
    
    metrics_dataframe = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "Cross-validation mean", "Cross-validation std"], columns=['q_kernel_training'])
    print('Classification Report: \n')
    print(classification_report(y_test,y_pred))
            
    return metrics_dataframe
    
