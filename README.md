# HephAIstos for Running Machine Learning on CPUs, GPUs and QPUs

We developed hephAIstos which is a Python open-source framework to run machine learning pipelines on CPUs, GPUs and quantum computing (QPUs). HephAIstos is distributed under the Apache License, Version 2.0. HephAIstos uses different frameworks such as Scikit-Learn, Keras on TensorFlow, Qiskit as well as homemade code. We can create a pipeline thanks to a Python function with parameters. We can also use specific routines. Contributions to improve the framework are more than welcome. 

We can find it on GitHub: https://github.com/xaviervasques/hephaistos.git

## Installation 

First, you can install hephAIstos by cloning it from GitHub. You can download it or type in your terminal:

`git clone https://github.com/xaviervasques/hephaistos.git`

Then, move to the Hephaistos directory and install requirements:

`pip install -r requirements.py`

HephAIstos has the following dependencies: 

* Python 
* joblib 
* numpy
* scipy
* pandas
* sklearn
* category_encoders
* hashlib
* matplotlib
* tensorflow
* qiskit
* qiskit_machine_learning

Datasets are available in the package. 

The following techniques are available in HephAIstos:

* Features Rescaling: StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Unit Vector Normalization, Log transformation, Square Root transformation, Reciprocal transformation, Box-Cox, Yeo-Johnson, Quantile Gaussian, and Quantile Uniform. 
* Categorical Data Encoding: Ordinal Encoding, One Hot Encoding, Label Encoding, Helmert Encoding, Binary Encoding, Frequency Encoding, Mean Encoding, Sum Encoding, Weigh to Evidence Encoding, Probability Ratio Encoding, Hashing Encoding, Backward Difference Encoding, Leave One Out Encoding, James-Stein Encoding, M-estimator. 
* Time-Related Feature Engineering: Time Split (year, month, seconds, …), Lag, Rolling Window, Expending Window.
* Missing Values: Row/Column Removal, Statistical Imputation (Mean, Median, Mode), Linear Interpolation, Multivariate Imputation by Chained Equation (MICE) Imputation, KNN Imputation.
* Feature Extraction: Principal Component Analysis, Independent Component Analysis, Linear Discriminant Analysis, Locally Linear Embedding, t-distributed Stochastic Neighbor Embedding, Manifold Learning Techniques
* Feature Selection: Filter methods (Variance Threshold, Statistical Tests, Chi-Square Test, ANOVA F-value, Pearson correlation coefficient), Wrapper methods (Forward Stepwise Selection, Backward Elimination, Exhaustive Feature Selection) and Embedded methods (Least Absolute Shrinkage and Selection Operator, Ridge Regression, Elastic Net, Regularization embedded into ML algorithms, Tree-based Feature Importance, Permutation Feature Importance)
* Classification algorithms running on CPUs: Support Vector Machine with linear, radial basis function, sigmoid and polynomial kernel functions (svm_linear,  svm_rbf, svm_sigmoid, svm_poly), Multinomial Logistic Regression (logistic_regression), Linear Discriminant Analysis (lda), Quadratic Qiscrimant Analysis (qda), Gaussian Naive Bayes (gnb), Multinomial Naive Bayes (mnb), K-Neighbors Naive Bayes (kneighbors), Stochastic Gradient Descent (sgd), Nearest Centroid Classifier (nearest_centroid), Decision Tree Classifier (decision_tree), Rendom Forest Classifier (random_forest), Extremely Randomized Trees (extra_trees), Multi-layer Perceptron Classifier (mlp_neural_network), Multi-layer Perceptron Classifier to run automatically different hyperparameters combinations and return the best result (mlp_neural_network_auto)
* Classification algorithms running on GPUs: Logistic Regression (gpu_logistic_regression), Multi-Layer perceptron (gpu_mlp), Recurrent Neural Network (gpu_rnn), 2D Convolutional Neural Network (conv2d)
* Classification algorithms running on QPUs: q_kernel_zz, q_kernel_default, q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_training, q_kernel_8_pegasos, q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos, q_kernel_default_pegasos
* Regression algorithms running on CPUs: Linear Regression (linear_regression), SVR with linear kernel (svr_linear), SVR with rbf kernel (svr_rbf), SVR with sigmoid kernel (svr_sigmoid), SVR with polynomial kernel (svr_poly), Multi-layer Perceptron for regression (mlp_regression), Multi-layer Perceptron to run automatically different hyperparameters combinations and return the best result (mlp_auto_regression)
* Regression algorithms running on GPUs: Linear Regression (gpu_linear_regression)

## HephAIstos function

To run a machine learning pipeline with hephAIstos we can use the Python function `ml_pipeline_function` composed of user defined parameters. This part aims at explaining all options of this `ml_pipeline_function`. 

The first parameter is mandatory as we need to give a DataFrame with a variable to predict that we call "Target". 

The rest of the parameters are optional, meaning that we can ignore them, and depends on what we need to run. Let’s say we provide a DataFrame defined as df, then we can apply the following options:

* __Save results__
  * `output_folder`: If you want to save figures, results, inference models to an output folder just set the path of an output folder where you want to save the results of the pipeline such as the metrics accuracy of the models in a .csv. 
  * Example: 

    ```python
  
    from ml_pipeline_function import ml_pipeline_function

    # Import dataset
    from data.datasets import neurons
    df = neurons()

    # Run ML Pipeline
    ml_pipeline_function(df, output_folder = './Outputs/')

    ```
  
  
* __Handle missing data__
  * `missing_method`: You can select different methodologies (Options: "row_removal", "column_removal", "stats_imputation_mean", "stats_imputation_median",     
  "stats_imputation_mode", "linear_interpolation", "mice", "knn". 
  * Example: 
  
    ```python
  
    from ml_pipeline_function import ml_pipeline_function

    # Import dataset
    from data.datasets import neurons
    df = neurons()

    ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal')

    ```

*	Split the data into training and testing datasets
  * `test_size`: If the dataset is not a time series dataset, we can set the amount of data we want for testing purposes. For example, if test_size = 0.2, 
  it means that we take 20% of the data for testing 
  * Example:
  
    ```python
  
    from ml_pipeline_function import ml_pipeline_function

    # Import dataset
    from data.datasets import neurons
    df = neurons()

    ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2)
  
    ```
  
  * test_time_size: If the dataset is a time series dataset, we do not use test_size but test_time_size instead. If you choose test_time_size = 1000, it    
  will take the last 1000 values of the dataset for testing.
  * time_feature_name: Name of the feature containing the time series
  * time_split: Used to split the time variable by year, month, minutes, seconds etc. as described in                           
  https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html
  Available options: 'year', 'month', 'hour', 'minute', 'second'
  * time_format: The strftime to parse time, e.g. "%d/%m/%Y". See strftime documentation for more information and the different options:                       
  https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
  For example if the data is as follows '1981-7-1 15:44:31', a format would be "%Y-%d-%m %H:%M:%S"
  * Example:

    ```python
  
    from ml_pipeline_function import ml_pipeline_function

    # Import Dataset
    from data.datasets import DailyDelhiClimateTrain
    df = DailyDelhiClimateTrain()
    df = df.rename(columns={"meantemp": "Target"})

    # Run ML Pipeline
    ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_time_size = 365, time_feature_name = 'date', time_format =    
    "%Y-%m-%d", time_split = ['year','month','day'])

    ```
  
* __Time series transformation__
  * time_transformation: if we want to transform time series data, we can use different techniques such as lag, rolling window, or expending window. For 
  example, if we want to use lag, we just need to set the time_transformation as follows: time_transformation = ‘lag’
  * If lag is selected, we need to add the following parameters:
    * number_of_lags: An integer defining the number of lags we want 
    * lagged_features: Select features we want to apply lag
    * lag_aggregation: Select the aggregation method. For aggregation, the following options are available: "min", "max", "mean", "std" or “no”. Several 
    options can be selected at the same time.
  * If rolling_window is selected:
    * window_size:  An integer indicating the size of the window
    * rolling_features: To select the features we want to apply rolling window
  * If expending_window is selected:
    * expending_window_size: An integer indicating the size of the window
    *expending_features: To select the features we want to apply expending rolling window.
  * Example:
    
      ```python
    
      from ml_pipeline_function import ml_pipeline_function

      from data.datasets import DailyDelhiClimateTrain
      df = DailyDelhiClimateTrain()
      df = df.rename(columns={"meantemp": "Target"})

      # Run ML Pipeline
      ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_time_size = 365, time_feature_name = 'date', time_format 
      = "%Y-%m-%d", time_split = ['year','month','day'], time_transformation='lag',number_of_lags=2, lagged_features = ['wind_speed', 'meanpressure'], 
      lag_aggregation = ['min', 'mean'])

      ```
    
* __Categorical data__
  * categorical: If the dataset is composed of categorical data that are labelled with text, we can select data encoding methods. The following options are available: ‘ordinal_encoding’, ‘one_hot_encoding’, ‘label_encoding’, ‘helmert_encoding’, ‘binary_encoding’, ‘frequency_encoding’, ‘mean_encoding’, ‘sum_encoding’, ‘weightofevidence_encoding’, ‘probability_ratio_encoding’, ‘hashing_encoding’, ‘backward_difference_encoding’, ‘leave_one_out_encoding’, ‘james_stein_encoding’, ‘m_estimator_encoding’. Different encoding methods can be combined. 
  * We need to select the features that we want to encode with the specified method. For this, we indicate the features we want to encode for each method:
    * features_ordinal
    * features_one_hot
    * features_label
    * features_helmert
    * features_binary
    * features_frequency
    * features_mean
    * features_sum
    * features_weight
    * features_proba_ratio
    * features_hashing
    * features_backward
    * features_leave_one_out
    * features_james_stein
    * features_m
  * Example:

     ```python
    
     from ml_pipeline_function import ml_pipeline_function
     import pandas as pd
     import numpy as np

     from data.datasets import insurance
     df = insurance()
     df = df.rename(columns={"charges": "Target"})

     # Run ML Pipeline
     ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
     ['binary_encoding','label_encoding'], features_binary = ['smoker','sex'], features_label = ['region'])

     ```

* __Data rescaling__
  * rescaling: Include a data rescaling method. The following options are available.
    * standard_scaler
    * minmax_scaler
    * maxabs_scaler
    * robust_scaler
    * normalizer
    * log_transformation
    * square_root_transformation
    * reciprocal_transformation
    * box_cox
    * yeo_johnson
    * quantile_gaussian 
    * quantile_uniform 
  * Example:

     ```python
    
     from ml_pipeline_function import ml_pipeline_function

     # Import Data
     from data.datasets import neurons
     df = neurons()

     # Run ML Pipeline
     ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
     ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler')

     ```
* __Features extraction__
  * features_extraction: Select the features extraction method. The following options are available.
    * pca
    * ica
    * icawithpca
    * lda_extraction
    * random_projection
    * truncatedSVD
    * isomap
    * standard_lle
    * modified_lle
    * hessian_lle
    * ltsa_lle
    * mds
    * spectral
    * tsne 
    * nca
  * number_components: Number of principal components we want to keep for PCA, ICA, LDA, etc. 
    * n_neighbors: Number of neighbors to consider for Manifold Learning techniques. 
  * Example:

      ```python
    
      from ml_pipeline_function import ml_pipeline_function

      # Import Data
      from data.datasets import neurons
      df = neurons()

      # Run ML Pipeline
      ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
      ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', features_extraction = 'pca', number_components = 2)

      ```
    
* __Features selection__
  * feature_selection: Here we can select a feature selection method (Filter, Wrapper and Embedded):
    * Filter options:
      * variance_threshold to apply variance threshold. If we choose this option, we need also to indicate the features we want to process 
      (features_to_process= ['feature_1', 'feature_2', …]) and the threshold (var_threshold=0 or any number)
      * chi2: Perform a chi-square test to the samples and retrieve only the k-best features. We can define k with the k_features parameter. 
      * anova_f_c: Create an SelectKBest object to select features with the k-best ANOVA F-Values for classification. We can define k with the k_features 
      parameter.
      * anova_f_r: Create an SelectKBest object to select features with the k-best ANOVA F-Values for regression. We can define k with the k_features 
      parameter.
      * pearson: The main idea for feature selection is to keep the variables that are highly correlated with the target and keep features that are 
      uncorrelated among themselves. The Pearson correlation coefficient between features is defined by cc_features and between features and the target by 
      cc_target.  
    * Examples:

       ```python
      
       ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
       ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', feature_selection = 'pearson', cc_features = 0.7, cc_target = 0.7)

       ```
      
       ```python
      
       ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
       ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', feature_selection = anova_f_c, k_features = 2)
      
       ```
    * Wrapper methods: The following options are available for feature_selection: ‘forward_stepwise’, ‘backward_elimination’, ‘exhaustive’.	
      * wrapper_classifier: In wrapper methods, we need to select a classifier or regressor. Here we can choose one from scikit-learn such as 
      KneighborsClassifier(), RandomForestClassifier, LinearRegression etc. and apply it to forward stepwise (forward_stepwise), backward elimination 
      (backward_elimination) or exhaustive (exhaustive) methods.
      * min_features and max_features are attributes for exhaustive to specify the minimum and maximum number of features we want in the combination
    * Example:

       ```python
      
       from ml_pipeline_function import ml_pipeline_function
       from sklearn.neighbors import KNeighborsClassifier

       from data.datasets import breastcancer
       df = breastcancer()
       df = df.drop(["id"], axis = 1)

       # Run ML Pipeline
       ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
       ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', feature_selection = 'backward_elimination', wrapper_classifier = 
       KNeighborsClassifier())

       ```
    * Embedded methods:
      * feature_selection: We can select several methods.
        * lasso: If we choose lasso, we need to add the alpha parameter (lasso_alpha)
        * feat_reg_ml: Selecting features with regularization embedded into machine learning algorithms. We need to select the machine learning algorithms 
        (in scikit-learn) by setting the parameter ml_penalty:
          * embedded_linear_regression
          * embedded_logistic_regression
          * embedded_decision_tree_regressor
          * embedded_decision_tree_classifier
          * embedded_random_forest_regressor
          * embedded_random_forest_classifier
          * embedded_permutation_regression
          * embedded_permutation_classification
          * embedded_xgboost_regression
          * embedded_xgboost_classification
      * Example:

       ```python
      
       from ml_pipeline_function import ml_pipeline_function
       from sklearn.svm import LinearSVC

       from data.datasets import breastcancer
       df = breastcancer()
       df = df.drop(["id"], axis = 1)

       # Run ML Pipeline
       ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
       ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', feature_selection = 'feat_reg_ml', ml_penalty = LinearSVC(C=0.05,    
       penalty='l1', dual=False, max_iter = 5000))

       ```
      
* __Classification algorithms__
  * Classification_algorithms: Classification algorithms used only with CPUs
    * svm_linear,
    * svm_rbf,
    * svm_sigmoid,
    * svm_poly,
    * logistic_regression,
    * lda,
    * qda,
    * gnb,
    * mnb,
    * kneighbors,
      * For k-neighbors, we need to add an additional parameter which is the number of neighbors (n_neighbors)
    * sgd,
    * nearest_centroid,
    * decision_tree,
    * random_forest,
      * For random_forest, we can optionally add the number of estimators (n_estimators_forest)
    * extra_trees,
      * For extra_trees, we add the number of estimators (n_estimators_forest)
    * mlp_neural_network,
      * The following parameters are available: max_iter, hidden_layer_sizes, activation, solver, alpha, learning_rate, learning_rate_init.
      * max_iter: Maximum number of iterations (default= 200)
      * hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
      * mlp_activation: Activation function for the hidden layer ('identity', 'logistic', 'relu', 'softmax', 'tanh'). default=’relu’
      * solver: The solver for weight optimization (‘lbfgs’, ‘sgd’, ‘adam’). default=’adam’
      * alpha: Strength of the L2 regularization term (default=0.0001)
      * mlp_learning_rate: Learning rate schedule for weight updates (‘constant’, ‘invscaling’, ‘adaptive’). default='constant'
      * learning_rate_init: The initial learning rate used (for sgd or adam). It controls the step-size in updating the weights.
    * mlp_neural_network_auto

  * For each classification algorithms, we also need to add the number of k-folds for cross-validation (cv). 
  * Example:

    ```python

     from ml_pipeline_function import ml_pipeline_function

     from data.datasets import breastcancer
     df = breastcancer()
     df = df.drop(["id"], axis = 1)

     # Run ML Pipeline
     ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
     ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', classification_algorithms=['svm_rbf','lda', 'random_forest'], 
     n_estimators_forest = 100, cv = 5)

    ```

This will print the steps of the processes and give the metrics of our models such as:

<img width="454" alt="image" src="https://user-images.githubusercontent.com/18941775/209166446-7447cd12-dd7f-47be-91c2-2126b3539719.png">

 * Classification algorithms that use GPUs
   * gpu_logistic_regression: we need to add parameters to use with gpu_logistic_regression:
   * gpu_logistic_optimizer : The model optimizers such as stochastic gradient descent (SGD(learning_rate = 1e-2)), adam ('adam') or RMSprop ('RMSprop').
   * gpu_logistic_loss: The loss functions such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class       
   logarithmic loss ('categorical_crossentropy').
   * gpu_logistic_epochs : Number of epochs
   * gpu_mlp: we need to add parameters to use gpu_mlp:
   * gpu_mlp_optimizer : The model optimizers such as stochastic gradient descent (SGD(learning_rate = 1e-2)), adam ('adam') or RMSprop ('RMSprop').
   * gpu_mlp_activation: The activation functions such as softmax, sigmoid, linear or tanh.
   * gpu_mlp_loss: The loss functions such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class  
   logarithmic loss ('categorical_crossentropy').
   * gpu_mlp_epochs : Number of epochs
   * gpu_rnn: Recurrent Neural Network for classification. We need to set the following parameters: 
   * rnn_units: Positive integer, dimensionality of the output space.
   * rnn_activation:  Activation function to use (softmax, sigmoid, linear or tanh)
   * rnn_optimizer: Optimizer (adam, sgd, RMSprop)
   * rnn_loss: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic 
   loss ('categorical_crossentropy').
   * rnn_epochs: Number (Integer) of epochs
 * Example:

   ```python
   
    from ml_pipeline_function import ml_pipeline_function

    from data.datasets import breastcancer
    df = breastcancer()
    df = df.drop(["id"], axis = 1)

    # Run ML Pipeline
    ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
    ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', classification_algorithms=['svm_rbf','lda', 'random_forest', 
    'gpu_logistic_regression'], n_estimators_forest = 100, gpu_logistic_activation = 'adam', gpu_logistic_optimizer = 'adam', gpu_logistic_epochs = 50, cv 
    = 5)

   ```
This will print the steps of the processes and give the metrics of our models such as:

<img width="454" alt="image" src="https://user-images.githubusercontent.com/18941775/209167611-1461427e-779b-4f4a-b5ca-2ea2e51c13e6.png">

 * Another example with SGD optimizer:

   ```python
   
     from ml_pipeline_function import ml_pipeline_function
     from tensorflow.keras.optimizers import SGD

     from data.datasets import breastcancer
     df = breastcancer()
     df = df.drop(["id"], axis = 1)

     # Run ML Pipeline
    ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
    ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', classification_algorithms=['svm_rbf','lda', 'random_forest', 
    'gpu_logistic_regression'], n_estimators_forest = 100, gpu_logistic_optimizer = SGD(learning_rate = 0.001), gpu_logistic_epochs = 50, cv = 5)
    
    ```
    
This gives the following metrics at the end:

<img width="454" alt="image" src="https://user-images.githubusercontent.com/18941775/209168066-afd4cde7-4b6f-4086-b6e3-0895927e13e1.png">

  * Classification algorithms that use QPUs
   * We use the encoding a default encoding function (q_kernel_default) and five encoding functions presented by Suzuki et al. (q_kernel_8, q_kernel_9, 
   q_kernel_10, q_kernel_11, q_kernel_12) and apply SVC from scikit-learn.
   * We also use a default encoding function (q_kernel_default_pegasos) and five encoding functions presented by Suzuki et al. (q_kernel_8_pegasos, 
   q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos) and apply Pegasos algorithm from Shalev-Shwartz.
   * We can also set up the QuantumKernel class to calculate a kernel matrix using the ZZFeatureMap (q_kernel_zz) with SVC from scikit-learn or pegasos 
   algorithm (q_kernel_zz_pegasos)
   * Algorithms we can select: q_kernel_default, q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_default_pegasos, 
   q_kernel_8_pegasos, q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos, q_kernel_zz_pegasos
   * Neural Networks are also available: q_samplerqnn, q_estimatorqnn, q_vqc
   * We also use q_kernel_training. It is also possible to train a quantum kernel with Quantum Kernel Alignment (QKA) that iteratively adapts a 
   parametrized quantum kernel to a dataset and converging to the maximum SVM margin at the same time. To implement it, we prepare the dataset as usual 
   and define the quantum feature map. Then, we will use QuantumKernelTrained.fit method to train the kernel parameters and pass it to a machine learning 
   model. Here we need to also adapt several parameters that are in the code (see classification_qpu.py in the classification folder) such as: 

   * Setup of the optimizer:

     ```python
    
       spsa_opt = SPSA(maxiter=10, callback=cb_qkt.callback, learning_rate=0.05, perturbation=0.05)

     ```
   * Rotational layer to train and the number of circuits.

     ```python
    
      # Rotational layer to train. We rotate each qubit the same amount.
      from qiskit.circuit import ParameterVector
      user_params = ParameterVector("θ", 1)
      fm0 = QuantumCircuit(2) # Number of circuits 
      # Add more if necessary
      fm0.ry(user_params[0], 0)
      fm0.ry(user_params[0], 1)

     ```
   * Inputs for running quantum algorithms:
    * reps= number of times the feature map circuit is repeated,
    * ibm_account = None,
    * quantum_backend: Depending on your credentials
     * ibmq_qasm_simulator
     * ibmq_armonk
     * ibmq_santiago
     * ibmq_bogota
     * ibmq_lima
     * ibmq_belem
     * ibmq_quito
     * simulator_statevector
     * simulator_mps
     * simulator_extended_stabilizer
     * simulator_stabilizer
     * ibmq_manila
   * multiclass: We can use ‘OneVsRestClassifier’, ‘OneVsOneClassifier’, ‘svc’ if we want to pass our quantum kernel to SVC from scikit-learn or ‘None’ if 
   we want to use QSVC from Qiskit. 

   * For pegasos algorithms:
    * n_steps = number of steps performed during the training procedure
    * C = regularization parameter
    * Example:

     ```python
     
       # Run ML Pipeline
       ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = ['label_encoding'], 
       features_label = ['Target'], rescaling = 'standard_scaler',features_extraction = 'pca', classification_algorithms=['svm_linear'], number_components 
       = 2, cv = 5, quantum_algorithms = ['q_kernel_default', 'q_kernel_zz', 'q_kernel_8','q_kernel_9','q_kernel_10','q_kernel_11','q_kernel_12'], reps = 
       2, ibm_account = YOUR API, quantum_backend = 'qasm_simulator')

     ```
We can also choose ‘least_busy’ as quantum_backend option in order to execute the algorithms on a chip that as the lower number of jobs in the queue:

```python

quantum_backend = 'least_busy'

```

 * Regression algorithms
   * Regression algorithms used only with CPUs
    * linear_regression,
    * svr_linear,
    * svr_rbf,
    * svr_sigmoid,
    * svr_poly,
    * mlp_regression,
    * mlp_auto_regression.
   * Regression algorithms that use GPUs if available
    * gpu_linear_regression : Linear Regression using SGD optimizer. As for classification, we need to add few parameters:
     * gpu_linear_activation:  'linear'
     * gpu_linear_epochs: An integer to define the number of epochs
     * gpu_linear_learning_rate: learning rate for the SGD optimizer
     * gpu_linear_loss: The loss functions such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class 
    logarithmic loss ('categorical_crossentropy').
    * gpu_mlp_regression:    Multi-layer perceptron neural network using GPUs for regression with the following parameters to set:
     * gpu_mlp_epochs_r: The number of epochs with an integer 
     * gpu_mlp_activation_r: The activation function such as softmax, sigmoid, linear or tanh. 
     * The optimizer chosen is ‘adam’. Note that no activation function is used for the output layer because it is a regression. We use mean_squared_error 
     for the loss function.
   * gpu_rnn_regression: Recurrent Neural Network for regression. We need to set the following parameters: 
    * rnn_units: Positive integer, dimensionality of the output space.
    * rnn_activation:  Activation function to use (softmax, sigmoid, linear or tanh)
    * rnn_optimizer: Optimizer (adam, sgd, RMSprop)
    * rnn_loss: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic 
    loss ('categorical_crossentropy').
    * rnn_epochs: Number (Integer) of epochs
 * Example:

   ```python
   
     from ml_pipeline_function import ml_pipeline_function
     from tensorflow.keras.optimizers import SGD

     from data.datasets import breastcancer
     df = breastcancer()
     df = df.drop(["id"], axis = 1)

     # Run ML Pipeline
     ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, categorical = 
     ['label_encoding'],features_label = ['Target'], rescaling = 'standard_scaler', regression_algorithms=['linear_regression','svr_linear', 'svr_rbf', 
     'gpu_linear_regression'], gpu_linear_epochs = 50, gpu_linear_activation = 'linear', gpu_linear_learning_rate = 0.01, gpu_linear_loss = ‘mse’)

   ```
This will print the steps of the processes and give the metrics of our models (the data used not really adapted) such as:

<img width="364" alt="image" src="https://user-images.githubusercontent.com/18941775/209170261-adde4737-c33a-437d-8b3f-b153c3a9cd99.png">

<img width="364" alt="image" src="https://user-images.githubusercontent.com/18941775/209170294-c22a4803-04ae-43db-b9a3-13966a18eead.png">

Another example with RNN:

   ```python
   
     from ml_pipeline_function import ml_pipeline_function
     import pandas as pd

     # Load data
     DailyDelhiClimateTrain = './data/datasets/DailyDelhiClimateTrain.csv'
     df = pd.read_csv(DailyDelhiClimateTrain, delimiter=',')

     # define time format
     df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')

     # create a DataFrame with new columns (year, month and day)
     df['year']=df['date'].dt.year
     df['month']=df['date'].dt.month
     df['day']=df['date'].dt.day

     # Delete column 'date'
     df.drop('date', inplace=True, axis=1)

     # Rename colunm meantemp to Target
     df = df.rename(columns={"meantemp": "Target"})

     # Drop row having at least 1 missing value
     df = df.dropna()

     # Run ML Pipeline
     ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal', test_size = 0.2, rescaling = 'standard_scaler', 
     regression_algorithms=['gpu_rnn_regression'], rnn_units = 500, rnn_activation = 'tanh' , rnn_optimizer = 'RMSprop', rnn_loss = 'mse', rnn_epochs = 
     50)

   ```

  * Convolutional Neural Networks
   * conv2d: 2D convolutional neural network (CNN) using GPUs if they are available. The parameters are the following: 
    * conv_kernel_size: The kernel_size is the size of the filter matrix for the convolution (conv_kernel_size x conv_kernel_size).
    * conv_activation:  Activation function to use (softmax, sigmoid, linear, relu or tanh)
    * conv_optimizer: Optimizer (adam, sgd, RMSprop)
    * conv_loss: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy') or the multi-class logarithmic 
    loss ('categorical_crossentropy').
    * conv_epochs: Number (Integer) of epochs
  * Example:

    ```python
    
      from ml_pipeline_function import ml_pipeline_function
      import pandas as pd

      import tensorflow as tf
      from keras.datasets import mnist
      from tensorflow.keras.utils import to_categorical

      df = mnist.load_data()
      (X, y), (_,_) = mnist.load_data()
      (X_train, y_train), (X_test, y_test) = df
                
      # Here we reshape the data to fit model with X_train.shape[0] images for training, image size is X_train.shape[1] x X_train.shape[2]
      # 1 means that the image is greyscale.
      X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
      X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
      X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)

      ml_pipeline_function(df, X, y, X_train, y_train, X_test, y_test, output_folder = './Outputs/', convolutional=['conv2d'], conv_activation='relu', 
      conv_kernel_size = 3, conv_optimizer = 'adam', conv_loss='categorical_crossentropy', conv_epochs=1)

    ```



