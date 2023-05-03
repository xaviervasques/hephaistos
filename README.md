# HephAIstos: A Framework for Running Machine Learning on CPUs, GPUs, and QPUs

HephAIstos is an open-source Python framework for running machine learning pipelines on CPUs, GPUs, and quantum computing (QPUs). Distributed under the Apache License, Version 2.0, HephAIstos utilizes various frameworks such as Scikit-Learn, Keras on TensorFlow, Qiskit, and custom code. You can create a pipeline using a Python function with parameters or leverage specific routines. Contributions to improve the framework are highly encouraged.

Find it on GitHub: [https://github.com/xaviervasques/hephaistos.git](https://github.com/xaviervasques/hephaistos.git)

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Techniques](#techniques)
  * [Feature Rescaling](#feature-rescaling)
  * [Categorical Data Encoding](#categorical-data-encoding)
  * [Time-Related Feature Engineering](#time-related-feature-engineering)
  * [Missing Values](#missing-values)
  * [Feature Extraction](#feature-extraction)
  * [Feature Selection](#feature-selection)
  * [Classification Algorithms for CPUs](#classification-algorithms-for-cpus)
  * [Classification Algorithms for GPUs](#classification-algorithms-for-gpus)
  * [Classification Algorithms for QPUs](#classification-algorithms-for-qpus)
  * [Regression Algorithms for CPUs](#regression-algorithms-for-cpus)
  * [Regression Algorithms for GPUs](#regression-algorithms-for-gpus)
  * [QPU-based Classification Algorithms](#qpu-based-classification-algorithms)
  * [Convolutional Neural Networks](#convolutional-neural-networks)
  
## Installation

To install HephAIstos, clone it from GitHub by downloading or typing the following command in your terminal:

```
git clone https://github.com/xaviervasques/hephaistos.git
```

Next, navigate to the Hephaistos directory and install the requirements:

```
pip install -r requirements.py
```

## Dependencies

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

Datasets are included in the package. 

## Techniques

### Feature Rescaling

StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Unit Vector Normalization, Log transformation, Square Root transformation, Reciprocal transformation, Box-Cox, Yeo-Johnson, Quantile Gaussian, and Quantile Uniform.

### Categorical Data Encoding

Ordinal Encoding, One Hot Encoding, Label Encoding, Helmert Encoding, Binary Encoding, Frequency Encoding, Mean Encoding, Sum Encoding, Weight of Evidence Encoding, Probability Ratio Encoding, Hashing Encoding, Backward Difference Encoding, Leave One Out Encoding, James-Stein Encoding, M-estimator.

### Time-Related Feature Engineering

Time Split (year, month, seconds, …), Lag, Rolling Window, Expanding Window.

### Missing Values

Row/Column Removal, Statistical Imputation (Mean, Median, Mode), Linear Interpolation, Multivariate Imputation by Chained Equation (MICE) Imputation, KNN Imputation.

### Feature Extraction

Principal Component Analysis, Independent Component Analysis, Linear Discriminant Analysis, Locally Linear Embedding, t-distributed Stochastic Neighbor Embedding, Manifold Learning Techniques.

### Feature Selection

Filter methods (Variance Threshold, Statistical Tests, Chi-Square Test, ANOVA F-value, Pearson correlation coefficient), Wrapper methods (Forward Stepwise Selection, Backward Elimination, Exhaustive Feature Selection), and Embedded methods (Least Absolute Shrinkage and Selection Operator, Ridge Regression, Elastic Net, Regularization embedded into ML algorithms, Tree-based Feature Importance, Permutation Feature Importance).

### Classification Algorithms for CPUs

Support Vector Machine with linear, radial basis function, sigmoid, and polynomial kernel functions (svm_linear, svm_rbf, svm_sigmoid, svm_poly), Multinomial Logistic Regression (logistic_regression), Linear Discriminant Analysis (lda), Quadratic Discriminant Analysis (qda), Gaussian Naive Bayes (gnb), Multinomial Naive Bayes (mnb), K-Neighbors Naive Bayes (kneighbors), Stochastic Gradient Descent (sgd), Nearest Centroid Classifier (nearest_centroid), Decision Tree Classifier (decision_tree), Random Forest Classifier (random_forest), Extremely Randomized Trees (extra_trees), Multi-layer Perceptron Classifier (mlp_neural_network), Multi-layer Perceptron Classifier with automatic hyperparameter tuning (mlp_neural_network_auto).

### Classification Algorithms for GPUs

Logistic Regression (gpu_logistic_regression), Multi-Layer perceptron (gpu_mlp), Recurrent Neural Network (gpu_rnn), 2D Convolutional Neural Network (conv2d).

### Classification Algorithms for QPUs

q_kernel_zz, q_kernel_default, q_kernel_8, q_kernel_9, q_kernel_10, q_kernel_11, q_kernel_12, q_kernel_training, q_kernel_8_pegasos, q_kernel_9_pegasos, q_kernel_10_pegasos, q_kernel_11_pegasos, q_kernel_12_pegasos, q_kernel_default_pegasos.

### Regression Algorithms for CPUs

Linear Regression (linear_regression), SVR with linear kernel (svr_linear), SVR with rbf kernel (svr_rbf), SVR with sigmoid kernel (svr_sigmoid), SVR with polynomial kernel (svr_poly), Multi-layer Perceptron for regression (mlp_regression), Multi-layer Perceptron with automatic hyperparameter tuning for regression (mlp_auto_regression).

### Regression Algorithms for GPUs

Linear Regression (gpu_linear_regression), Multi-layer perceptron neural network using GPUs for regression (gpu_mlp_regression), Recurrent Neural Network for regression (gpu_rnn_regression).

### Convolutional Neural Networks

2D convolutional neural network (CNN) using GPUs if they are available (conv2d). 

## HephAIstos Function

To run a machine learning pipeline with HephAIstos, use the Python function `ml_pipeline_function`, which accepts user-defined parameters. This section explains all available options for the `ml_pipeline_function`.

The first parameter is mandatory, as it requires a DataFrame with a variable to predict, which we call "Target".

The rest of the parameters are optional, meaning you can ignore them, and depend on what you need to run. Assuming you provide a DataFrame defined as `df`, you can apply the following options:

### Save Results

* `output_folder`: If you want to save figures, results, inference models to an output folder, set the path of the output folder where you want to save the pipeline results, such as the model's accuracy metrics in a .csv file.
  * Example:

    ```python
    from ml_pipeline_function import ml_pipeline_function

    # Import dataset
    from data.datasets import neurons
    df = neurons()

    # Run ML Pipeline
    ml_pipeline_function(df, output_folder='./Outputs/')
    ```

### Handle Missing Data

* `missing_method`: You can select different methodologies (Options: "row_removal", "column_removal", "stats_imputation_mean", "stats_imputation_median", "stats_imputation_mode", "linear_interpolation", "mice", "knn").
  * Example:

    ```python
    from ml_pipeline_function import ml_pipeline_function

    # Import dataset
    from data.datasets import neurons
    df = neurons()

    ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal')
    ```

### Split the Data into Training and Testing Datasets

* `test_size`: If the dataset is not a time series dataset, you can set the amount of data you want for testing purposes. For example, if `test_size=0.2`, it means that 20% of the data is used for testing.
  * Example:

    ```python
    from ml_pipeline_function import ml_pipeline_function

    # Import dataset
    from data.datasets import neurons
    df = neurons()

    ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2)
    ```

* `test_time_size`: If the dataset is a time series dataset, use `test_time_size` instead of `test_size`. If you choose `test_time_size=1000`, it will take the last 1000 values of the dataset for testing.
* `time_feature_name`: Name of the feature containing the time series.
* `time_split`: Used to split the time variable by year, month, minutes, seconds, etc., as described in [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html). Available options: 'year', 'month', 'hour', 'minute', 'second'.
* `time_format`: The strftime to parse time, e.g., "%d/%m/%Y". See [strftime documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) for more information and different options. For example, if the data is '1981-7-1 15:44:31', the format would be "%Y-%d-%m %H:%M:%S".
  * Example:

    ```python
    from ml_pipeline_function import ml_pipeline_function

    # Import Dataset
    from data.datasets import DailyDelhiClimateTrain
    df = DailyDelhiClimateTrain()
    df = df.rename(columns={"meantemp": "Target"})

    # Run ML Pipeline
    ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_time_size=365, time_feature_name='date', time_format="%Y-%m-%d", time_split=['year', 'month', 'day'])
    ```
  
### Time Series Transformation

* `time_transformation`: If you want to transform time series data, you can use different techniques such as lag, rolling window, or expanding window. For example, if you want to use lag, you just need to set the `time_transformation` as follows: `time_transformation='lag'`.

* If `lag` is selected, you need to add the following parameters:
  * `number_of_lags`: An integer defining the number of lags you want.
  * `lagged_features`: Select features you want to apply lag.
  * `lag_aggregation`: Select the aggregation method. For aggregation, the following options are available: "min", "max", "mean", "std", or "no". Several options can be selected at the same time.

* If `rolling_window` is selected:
  * `window_size`: An integer indicating the size of the window.
  * `rolling_features`: Select the features you want to apply rolling window.

* If `expanding_window` is selected:
  * `expanding_window_size`: An integer indicating the size of the window.
  * `expanding_features`: Select the features you want to apply expanding rolling window.

* Example:

  ```python
  from ml_pipeline_function import ml_pipeline_function

  from data.datasets import DailyDelhiClimateTrain
  df = DailyDelhiClimateTrain()
  df = df.rename(columns={"meantemp": "Target"})

  # Run ML Pipeline
  ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_time_size=365, time_feature_name='date', time_format="%Y-%m-%d", time_split=['year', 'month', 'day'], time_transformation='lag', number_of_lags=2, lagged_features=['wind_speed', 'meanpressure'], lag_aggregation=['min', 'mean'])
  ```
    
### Categorical Data

* `categorical`: If the dataset is composed of categorical data that are labeled with text, you can select data encoding methods. The following options are available: 'ordinal_encoding', 'one_hot_encoding', 'label_encoding', 'helmert_encoding', 'binary_encoding', 'frequency_encoding', 'mean_encoding', 'sum_encoding', 'weightofevidence_encoding', 'probability_ratio_encoding', 'hashing_encoding', 'backward_difference_encoding', 'leave_one_out_encoding', 'james_stein_encoding', 'm_estimator_encoding'. Different encoding methods can be combined.

* You need to select the features that you want to encode with the specified method. For this, indicate the features you want to encode for each method:
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

  from data.datasets import insurance
  df = insurance()
  df = df.rename(columns={"charges": "Target"})

  # Run ML Pipeline
  ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['binary_encoding', 'label_encoding'], features_binary=['smoker', 'sex'], features_label=['region'])
  ```

### Data Rescaling

* `rescaling`: Include a data rescaling method. The following options are available.
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
  ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler')
  ```
  
### Feature Extraction

* `features_extraction`: Select the feature extraction method. The following options are available:
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

* `number_components`: Number of principal components you want to keep for PCA, ICA, LDA, etc.
* `n_neighbors`: Number of neighbors to consider for Manifold Learning techniques.

* Example:

  ```python
  from ml_pipeline_function import ml_pipeline_function

  # Import Data
  from data.datasets import neurons
  df = neurons()

  # Run ML Pipeline
  ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', features_extraction='pca', number_components=2)
  ```

This code snippet demonstrates how to use the `ml_pipeline_function` to run an ML pipeline with feature extraction using PCA. The dataset is first preprocessed by handling missing data, encoding categorical features, and rescaling the data. After that, PCA is applied to extract two principal components before the final model is fit and evaluated.
    
## Features Selection

The `ml_pipeline_function` allows you to select a feature selection method (Filter, Wrapper, and Embedded) to improve your machine learning model's performance. Different options and examples are provided below:

### Filter Methods

Filter options include:

- `variance_threshold`: Apply variance threshold. If you choose this option, you need to indicate the features to process (`features_to_process= ['feature_1', 'feature_2', …]`) and the threshold (`var_threshold=0` or any number).
- `chi2`: Perform a chi-square test on the samples and retrieve only the k-best features. You can define k with the `k_features` parameter.
- `anova_f_c`: Create a SelectKBest object to select features with the k-best ANOVA F-Values for classification. You can define k with the `k_features` parameter.
- `anova_f_r`: Create a SelectKBest object to select features with the k-best ANOVA F-Values for regression. You can define k with the `k_features` parameter.
- `pearson`: Keep variables highly correlated with the target and uncorrelated among themselves. Define the Pearson correlation coefficient between features with `cc_features` and between features and the target with `cc_target`.

Examples:

```python
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', feature_selection='pearson', cc_features=0.7, cc_target=0.7)
```

```python
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', feature_selection='anova_f_c', k_features=2)
```

### Wrapper Methods

Available `feature_selection` options for wrapper methods: `forward_stepwise`, `backward_elimination`, `exhaustive`.

- `wrapper_classifier`: Select a classifier or regressor from scikit-learn (e.g., KneighborsClassifier(), RandomForestClassifier, LinearRegression) and apply it to forward stepwise, backward elimination, or exhaustive methods.
- `min_features` and `max_features` specify the minimum and maximum number of features in the combination for exhaustive methods.

Example:

```python
from ml_pipeline_function import ml_pipeline_function
from sklearn.neighbors import KNeighborsClassifier
from data.datasets import breastcancer

df = breastcancer()
df = df.drop(["id"], axis=1)

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', feature_selection='backward_elimination', wrapper_classifier=KNeighborsClassifier())
```

### Embedded Methods

You can select several embedded methods for `feature_selection`:

- `lasso`: If you choose lasso, add the `alpha` parameter (`lasso_alpha`).
- `feat_reg_ml`: Select features with regularization embedded into machine learning algorithms. Select the machine learning algorithm (in scikit-learn) by setting the `ml_penalty` parameter:
  - embedded_linear_regression
  - embedded_logistic_regression
  - embedded_decision_tree_regressor
  - embedded_decision_tree_classifier
  - embedded_random_forest_regressor
  - embedded_random_forest_classifier
  - embedded_permutation_regression
  - embedded_permutation_classification
  - embedded_xgboost_regression
  - embedded_xgboost_classification

Example:

```python
from ml_pipeline_function import ml_pipeline_function
from sklearn.svm import LinearSVC
from data.datasets import breastcancer

df = breastcancer()
df = df.drop(["id"], axis=1)

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', feature_selection='feat_reg_ml', ml_penalty=LinearSVC(C=0.05, penalty='l1', dual=False, max_iter=5000))
```

In this example, we use the `feat_reg_ml` embedded feature selection method with a LinearSVC classifier. The LinearSVC classifier is specified with an L1 penalty, a C value of 0.05, and a maximum of 5000 iterations. The pipeline uses this classifier for feature selection and then proceeds with training and evaluation.


## Classification Algorithms

### CPU-based Classification Algorithms

Choose from the following classification algorithms that run on CPUs:

* `svm_linear`
* `svm_rbf`
* `svm_sigmoid`
* `svm_poly`
* `logistic_regression`
* `lda`
* `qda`
* `gnb`
* `mnb`
* `kneighbors`: Add an additional parameter, `n_neighbors`, to specify the number of neighbors.
* `sgd`
* `nearest_centroid`
* `decision_tree`
* `random_forest`: Optionally add the `n_estimators_forest` parameter to specify the number of estimators.
* `extra_trees`: Add the `n_estimators_forest` parameter to specify the number of estimators.
* `mlp_neural_network`: Set various parameters such as `max_iter`, `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`, and `learning_rate_init`.
* `mlp_neural_network_auto`

For each classification algorithm, add the number of k-folds for cross-validation (`cv`).

#### Example:

```python
from ml_pipeline_function import ml_pipeline_function
from data.datasets import breastcancer

df = breastcancer()
df = df.drop(["id"], axis=1)

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', classification_algorithms=['svm_rbf', 'lda', 'random_forest'], n_estimators_forest=100, cv=5)
```

### GPU-based Classification Algorithms

Choose from the following classification algorithms that run on GPUs:

* `gpu_logistic_regression`: Set parameters such as `gpu_logistic_optimizer`, `gpu_logistic_loss`, and `gpu_logistic_epochs`.
* `gpu_mlp`: Set parameters such as `gpu_mlp_optimizer`, `gpu_mlp_activation`, `gpu_mlp_loss`, and `gpu_mlp_epochs`.
* `gpu_rnn`: Set parameters such as `rnn_units`, `rnn_activation`, `rnn_optimizer`, `rnn_loss`, and `rnn_epochs`.

#### Example:

```python
from ml_pipeline_function import ml_pipeline_function
from data.datasets import breastcancer

df = breastcancer()
df = df.drop(["id"], axis=1)

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', classification_algorithms=['svm_rbf', 'lda', 'random_forest', 'gpu_logistic_regression'], n_estimators_forest=100, gpu_logistic_activation='adam', gpu_logistic_optimizer='adam', gpu_logistic_epochs=50, cv=5)
```

Running the pipeline will print the steps of the processes and provide metrics for the models, such as accuracy, precision, recall, and F1 score.

![image](https://user-images.githubusercontent.com/18941775/209166446-7447cd12-dd7f-47be-91c2-2126b3539719.png)

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

## QPU-based Classification Algorithms

Choose from the following classification algorithms that run on Quantum Processing Units (QPUs):

* Quantum Kernel methods: Select from the following algorithms:
  * `q_kernel_default`
  * `q_kernel_8`
  * `q_kernel_9`
  * `q_kernel_10`
  * `q_kernel_11`
  * `q_kernel_12`
  * `q_kernel_default_pegasos`
  * `q_kernel_8_pegasos`
  * `q_kernel_9_pegasos`
  * `q_kernel_10_pegasos`
  * `q_kernel_11_pegasos`
  * `q_kernel_12_pegasos`
  * `q_kernel_zz`
  * `q_kernel_zz_pegasos`
* Quantum Neural Networks: Select from the following algorithms:
  * `q_samplerqnn`
  * `q_estimatorqnn`
  * `q_vqc`
* Quantum Kernel Alignment (QKA) methods: Use `q_kernel_training`.

### Parameters for QPU-based Algorithms

* `reps`: Number of times the feature map circuit is repeated.
* `ibm_account`: Your IBM Quantum API key.
* `quantum_backend`: The backend to run the quantum algorithms on. Choose from:
  * `ibmq_qasm_simulator`
  * `ibmq_armonk`
  * `ibmq_santiago`
  * `ibmq_bogota`
  * `ibmq_lima`
  * `ibmq_belem`
  * `ibmq_quito`
  * `simulator_statevector`
  * `simulator_mps`
  * `simulator_extended_stabilizer`
  * `simulator_stabilizer`
  * `ibmq_manila`
* `multiclass`: Choose from:
  * `OneVsRestClassifier`
  * `OneVsOneClassifier`
  * `svc`: Pass the quantum kernel to SVC from scikit-learn.
  * `None`: Use QSVC from Qiskit.

### Additional Parameters for Pegasos Algorithms

* `n_steps`: Number of steps performed during the training procedure.
* `C`: Regularization parameter.

#### Example:

```python
from ml_pipeline_function import ml_pipeline_function
from data.datasets import breastcancer

df = breastcancer()
df = df.drop(["id"], axis=1)

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', features_extraction='pca', classification_algorithms=['svm_linear'], number_components=2, cv=5, quantum_algorithms=['q_kernel_default', 'q_kernel_zz', 'q_kernel_8', 'q_kernel_9', 'q_kernel_10', 'q_kernel_11', 'q_kernel_12'], reps=2, ibm_account=YOUR_API, quantum_backend='qasm_simulator')
```

To run the algorithm on the least busy quantum backend, set the `quantum_backend` parameter to `'least_busy'`.

```python
quantum_backend = 'least_busy'
```

## Regression Algorithms

The pipeline supports the following regression algorithms:

### CPU-based regression algorithms

* `linear_regression`
* `svr_linear`
* `svr_rbf`
* `svr_sigmoid`
* `svr_poly`
* `mlp_regression`
* `mlp_auto_regression`

### GPU-based regression algorithms (if available)

* `gpu_linear_regression`: Linear Regression using SGD optimizer. The following parameters can be configured:
  * `gpu_linear_activation`: `'linear'`
  * `gpu_linear_epochs`: An integer for the number of epochs
  * `gpu_linear_learning_rate`: Learning rate for the SGD optimizer
  * `gpu_linear_loss`: Loss functions such as `'mse'`, `'binary_crossentropy'`, or `'categorical_crossentropy'`.

* `gpu_mlp_regression`: Multi-layer perceptron neural network using GPUs for regression. The following parameters can be configured:
  * `gpu_mlp_epochs_r`: An integer for the number of epochs
  * `gpu_mlp_activation_r`: Activation function, e.g., `softmax`, `sigmoid`, `linear`, or `tanh`.

* `gpu_rnn_regression`: Recurrent Neural Network for regression. The following parameters can be configured:
  * `rnn_units`: Positive integer for the dimensionality of the output space
  * `rnn_activation`: Activation function, e.g., `softmax`, `sigmoid`, `linear`, or `tanh`
  * `rnn_optimizer`: Optimizer, e.g., `adam`, `sgd`, `RMSprop`
  * `rnn_loss`: Loss functions such as `'mse'`, `'binary_crossentropy'`, or `'categorical_crossentropy'`.
  * `rnn_epochs`: An integer for the number of epochs

## Usage Example

```python
from ml_pipeline_function import ml_pipeline_function
from data.datasets import breastcancer

df = breastcancer()
df = df.drop(["id"], axis=1)

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, categorical=['label_encoding'], features_label=['Target'], rescaling='standard_scaler', regression_algorithms=['linear_regression', 'svr_linear', 'svr_rbf', 'gpu_linear_regression'], gpu_linear_epochs=50, gpu_linear_activation='linear', gpu_linear_learning_rate=0.01, gpu_linear_loss='mse')
```

Running this example will print the steps of the processes and give the metrics of the models. Note that the data used might not be optimal for the chosen algorithms. The output will be similar to the following images:

![Example Output 1](https://user-images.githubusercontent.com/18941775/209170261-adde4737-c33a-437d-8b3f-b153c3a9cd99.png)
![Example Output 2](https://user-images.githubusercontent.com/18941775/209170294-c22a4803-04ae-43db-b9a3-13966a18eead.png)

## RNN Regression Example

In this example, we demonstrate how to use the pipeline for training a Recurrent Neural Network (RNN) regression model on the Daily Delhi Climate dataset.

```python
from ml_pipeline_function import ml_pipeline_function
import pandas as pd

# Load data
DailyDelhiClimateTrain = './data/datasets/DailyDelhiClimateTrain.csv'
df = pd.read_csv(DailyDelhiClimateTrain, delimiter=',')

# Define time format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Create a DataFrame with new columns (year, month, and day)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Delete column 'date'
df.drop('date', inplace=True, axis=1)

# Rename column meantemp to Target
df = df.rename(columns={"meantemp": "Target"})

# Drop rows with at least 1 missing value
df = df.dropna()

# Run ML Pipeline
ml_pipeline_function(df, output_folder='./Outputs/', missing_method='row_removal', test_size=0.2, rescaling='standard_scaler', regression_algorithms=['gpu_rnn_regression'], rnn_units=500, rnn_activation='tanh', rnn_optimizer='RMSprop', rnn_loss='mse', rnn_epochs=50)
```

Running this example will train the RNN regression model on the preprocessed dataset and display the model's performance metrics.

## Convolutional Neural Networks

### 2D Convolutional Neural Network (CNN)

`conv2d`: 2D convolutional neural network (CNN) using GPUs if they are available. The parameters are the following:

* `conv_kernel_size`: The size of the filter matrix for the convolution (conv_kernel_size x conv_kernel_size).
* `conv_activation`: Activation function to use (softmax, sigmoid, linear, relu, or tanh).
* `conv_optimizer`: Optimizer (adam, sgd, RMSprop).
* `conv_loss`: Loss function such as the mean squared error ('mse'), the binary logarithmic loss ('binary_crossentropy'), or the multi-class logarithmic loss ('categorical_crossentropy').
* `conv_epochs`: Number (integer) of epochs.

### Example

In this example, we train a 2D CNN on the MNIST dataset using the pipeline:

```python
from ml_pipeline_function import ml_pipeline_function
import pandas as pd
from keras.datasets import mnist

(X, y), (_, _) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the data to fit the model with X_train.shape[0] images for training,
# image size is X_train.shape[1] x X_train.shape[2], and 1 means the image is grayscale.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

ml_pipeline_function(
    X, y, X_train, y_train, X_test, y_test, output_folder='./Outputs/',
    convolutional=['conv2d'], conv_activation='relu',
    conv_kernel_size=3, conv_optimizer='adam',
    conv_loss='categorical_crossentropy', conv_epochs=1
)
```

Running this example will train the 2D CNN model on the preprocessed dataset and display the model's performance metrics.



