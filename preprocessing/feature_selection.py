#!/usr/bin/python3
# feature_selection.py
# Author: Xavier Vasques (Last update: 12/04/2022)

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
There are many ways to increase the amount of data we have at our disposal and increase the number of features. Often, we tend to increase the dimensionality of our data thinking that
it will augment the performance of our models. Increase the dimensionality can have some consequences such as increase complexity of the machine learning models, lead to overfitting,
increase computational time, and not always improve accuracy (misleading data) specially when we provide irrelevant features to linear algorithms such as linear and logistic
regression. The overfitting will appear when we inject more redundant data adding noise. With a similar idea than feature extraction, feature selection is a set of techniques to
reduce the input variables to our machine learning models by only using relevant features that contribute most to the prediction variable. Feature selection is different from feature
extraction because feature selection returns a subset of the original features whereas in feature extraction, we create new features from functions of the original features. Feature
selection allows to deal with simplest models and generalize better by choosing a small subset of the relevant features from the original features by removing noisy, redundant, and
irrelevant features. We can use both supervised and unsupervised techniques and we can classify features techniques by filter, wrapper, hybrid, and embedded methodologies. When we use
the filter approach is simply select features based on statistical measures such as variance threshold, chi-square test, Fisher score, correlation coefficient, or information gain.
The idea is to filter out irrelevant features and redundant information. With filter, we take the intrinsic properties of the features via univariate statistics. In univariate tests,
features are individuality considered which means that a feature can be rejected if it’s only informative when it’s combined with another one. The selected features are completely
independent of the model we will use later. We can compute a score by features and just select based on the score. In general, these methodologies require less computational time. The
wrapper methodology will use a predictive model to evaluate combinations of features and assign model performance scores. In other words, wrapper methods will evaluate on a specific
machine learning algorithm to find optimal features. This means that, at the opposite of filter methods, wrapper methods could need much more computation time if we are in the context
of high dimensional data. In embedded type feature selection, the selection of features is made within the model learning process itself. In fact, the features are selected during the
model building process in each iteration of model training phase. Regarding overfitting, if the number of observations is not enough, there is a risk of overfitting when using wrapper
methods.

"""

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

# Filter methods

def variance_threshold(data, threshold = None):
    '''
    One simple idea to select features is to remove all features which variance doesn’t meet some threshold. For instance, if we calculate the variance of a feature and the result is 0, it means that the feature has the same value in all the samples. In that case, no need to take it into account. The hypothesis we do here is that a feature with higher variance contain more significant information. The drawback of this method is that we do not consider the correlation or any relationship between features. This filter method is called variance threshold. In this method, we can set a threshold, such as a percentage, helping us identify the features to remove. Let’s say, we set the threshold at 80% and we apply this threshold to a feature containing 0s and 1s. If we have more than 80% of 0s or more than 80% of 1s, the feature will be removed.

    In the case of Boolean features (Bernoulli random variables), the variance is given by Var(X) = p(1-p) which means than the threshold (80%) is given by 0.8*(1-0.8).

    To select features with variance threshold, simply use scikit-learn library VarianceThreshold from sklearn.feature_selection.

    Inputs:
        - DataFrame
        - threshold
    
    '''

    print("\n")
    print("Selecting features with variance threshold: started")
    print("\n")

    from sklearn.feature_selection import VarianceThreshold
    # Get data column names
    columns_name = data.columns
    # By default, it removes all zero-variance features
    var = VarianceThreshold(threshold=threshold)
    data = var.fit_transform(data)
    # Create a new DataFrame
    df_data = pd.DataFrame(data = data, columns = columns_name)
    print(df_data.shape)
    
    print("\n")
    print("Variance Threshold: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data

def chi_square(X, y, k_features):
    '''
    Let’s detail a feature selection method under the filter category such as the chi-square (χ^2) test which is commonly used in statistics to test the statistical independence between two or more categorical variables. A and B are two independent events if:

    P(AB)=p(A)P(B)  or,equivalently P(A│B)=P(A)  and P(B│A)=P(B)

    To correctly apply chi-square, the data must be non-negative, sampled independently and be categorical such as Booleans or frequencies (greater than 5 as chi-square is sensitive to small frequencies).  It is not appropriate for continuous variables or to compare categorical with continuous variables. In statistics, if we want to test correlation (dependence) of two continuous variables, Pearson correlation is commonly used. When one variable is continuous and the other one is ordinal, Spearman’s rank correlation is appropriate. It is possible to apply chi-square on continuous variables after binning the variables.  The idea behind chi-square is to calculate a p-value, in order to identify a high correlation (low p-value) between two categorical variables or in other words the variables are dependent on each other (p<0.05). The p-value is calculated thanks to the chi-square score and degrees of freedom.

    In feature selection, we calculate chi-square between each feature and the target. We will select the desired number of features based on the best chi-square scores.
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - k_features: select the k-best features
    '''

    print("\n")
    print("Selecting features with chi-square: started")
    print("\n")
    
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    
    # Perform a chi-square test to the samples to retrieve only the k best features
    # Create and fit selector
    selector = SelectKBest(chi2, k = k_features)
    selector.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices = True)
    features_df_new = X.iloc[: , cols]

    print("\n")
    print("Chi-square: DataFrame")
    print("\n")
    print(features_df_new)

    return features_df_new

# ANOVA-F for classification tasks
def anova_f_c(X, y, k_features):
    '''
    If our features are continuous variables and our target vector categorical, the analysis of variance (ANOVA) F-value can be calculated to determine if the means of each group (features by the target vector) are significantly difference. When we run an ANOVA test or a regression analysis, we can compute an F statistic (F-test) to statistically assess the equality of means. F-test is like a t-test which determines if a single feature is statistically different, and the F-test will tell us if the means of three or more groups are different. In fact, when we apply ANOVA F-test to only two groups, F=t*t where t is the Student’s t statistic.  As the others statistical test, we will get a F value, a F critical value and a p-value.

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    '''

    print("\n")
    print("Selecting features with ANOVA-F for classification tasks: started")
    print("\n")
    
    # Select Features With Best ANOVA F-Values
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    # Create an SelectKBest object to select features with k best ANOVA F-Values (classification task)
    # Create and fit selector
    selector = SelectKBest(f_classif, k=k_features)
    selector.fit(X, y)

    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices = True)
    features_df_new = X.iloc[: , cols]

    print("\n")
    print("Analysis of Variance (ANOVA) F-value for classification tasks: DataFrame")
    print("\n")
    print(features_df_new)

    return features_df_new

# ANOVA-F for regression tasks
def anova_f_r(X, y, k_features):
    '''
    If our features are continuous variables and our target vector categorical, the analysis of variance (ANOVA) F-value can be calculated to determine if the means of each group (features by the target vector) are significantly difference. When we run an ANOVA test or a regression analysis, we can compute an F statistic (F-test) to statistically assess the equality of means. F-test is like a t-test which determines if a single feature is statistically different, and the F-test will tell us if the means of three or more groups are different. In fact, when we apply ANOVA F-test to only two groups, F=t*t where t is the Student’s t statistic.  As the others statistical test, we will get a F value, a F critical value and a p-value.

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    '''

    print("\n")
    print("Selecting features with ANOVA-F for regression tasks: started")
    print("\n")
    
    # Select Features With Best ANOVA F-Values
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    
    print("test")

    # Create an SelectKBest object to select features with k best ANOVA F-Values
    # Create and fit selector
    selector = SelectKBest(f_regression, k=k_features)
    selector.fit(X, y)

    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices = True)
    features_df_new = X.iloc[: , cols]

    print("\n")
    print("Analysis of Variance (ANOVA) F-value for regression tasks: DataFrame")
    print("\n")
    print(features_df_new)

    return features_df_new
    
def pearson(X, y, cc_features = None, cc_target = None):
    '''
    We can use the Pearson correlation coefficient (r) to help us measure the linear relationship between two or more variables or in other words how much we can predict one variable from another one. We use it for features selection with the idea that the variables to keep are those that are highly correlated with the target and uncorrelated among themselves. The Pearson correlation is a number between -1 and 1. If the value is close to 1, it indicates a strong positive correlation (if r = 1 there is a perfect linear correlation between two variables). If the value is close to -1 it means a strong negative correlation (r = -1 is a perfect inverse linear correlation) and values close to 0 indicates weak correlation (0 really means no linear correlation at all).

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - cc_features: Threshold of the coefficient of correlation. Keep all features that have a coefficient of correlation above cc_features between themselves
        - cc_target: Threshold of the coefficient of correlation. Keep all features that have a coefficient of correlation above cc_target with the target
    '''
    
    print("\n")
    print("Selecting features with Pearson: started")
    print("\n")

    # Create correlation matrix
    X = pd.DataFrame(X)
    corr_matrix = X.corr()
    print("Correlation Matrix:\n")
    print(corr_matrix)
    
    # Select upper triangle of correlation matrix
    #upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index of feature columns with correlation greater than the selected coefficient of correlation
    to_drop = [column for column in upper.columns if any(upper[column] > cc_features)]
    print("\n")
    print("Features to drop after Pearson selection:\n")
    print(to_drop)
    print("\n")
    
    # Drop Marked Features
    df_new = X.drop(to_drop, axis=1)
    
    # From the new generated dataset (df_new), we have now to set an absolute value as the threshold for selecting the features that are correlated with the target
    
    # Concatenate dataframes (X and the target y called diagnosis)
    df_y = pd.DataFrame(y,columns = ["Target"])
    df_all = pd.concat([df_new, df_y], axis=1)
    
    #Correlation with target variable
    cor = df_all.corr()
    cor_target = abs(cor["Target"])

    # Relevant features
    relevant_features = cor_target[cor_target > cc_target]

    print("Features to keep : \n")
    print(relevant_features)
    
    # List of features to keep
    list_features_to_keep=relevant_features.keys().to_list()
    list_features_to_keep.remove("Target")
    
    #print("\n")
    #print("Features to keep in the dataframe : \n")
    #print(list_features_to_keep)
    #print("\n")
 
    # Create a new DataFrame
    df_data = pd.DataFrame(data = df_all, columns = list_features_to_keep)
    
    print("\n")
    print("Pearson: New DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
# Wrapper Methods
    
def forward_stepwise(X, y, wrapper_classifier, k_features, cv):
    '''
    Forward stepwise selection is an iterative method which starts with no feature in the model and iteratively add the best performing feature, features by features, against the target (which best improves the model) until a feature does not improve the model performance or if we run out features. To follow this process, it is necessary to evaluate all features individually to select the one that provides the best performance to the algorithm based on a pre-set evaluation criteria. We will then combine the selected feature with another one which is selected based on the performance of the algorithm (we select the pair with the best score). As we said before regarding wrapper methods, forward stepwise selection can be computationally expensive since the selection procedure, usually called greedy, evaluates all possible feature combination (single, double, triple etc.). If the feature space is large, sometimes it is just not doable.

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - wrapper_classifier: select a scikit-learn classifier regressor such as RandomForestClassifier()
    
    '''

    print("\n")
    print("Selecting features with forward stepwise: started")
    print("\n")

    # Importing the necessary libraries for Forward stepwise feature selection
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    # Let's create our feature selector. SFS = Sequential Forward Selection
    sfs = SFS(wrapper_classifier,        # we can use any scikit-learn classifier or regressor
          k_features=k_features,           # number of features we want to keep
          forward=True,
          floating=False,
          scoring = 'accuracy',
          cv = cv,                         # Cross-validation
          n_jobs=-1)                              # Run the cross-validation on all our available CPU cores.

    # Fit the model
    sfs.fit(X, y)

    # Print selected features
    print("\n")
    print("Features to keep in the dataframe:")
    print(sfs.k_feature_names_)
    print("\n")
    
    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = sfs.k_feature_names_)
    
    print("\n")
    print("Forward Stepwise: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def backward_elimination(X, y, wrapper_classifier, k_features, cv):
    '''
    In backward elimination, at the contrary of forward stepwise, we start with the full model (all features including the independent features). The objective is to eliminate the least significant feature (the worst feature with the highest p-value > significant level) at each iteration until no improvement of the performance model is observed.

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - wrapper_classifier: select a scikit-learn classifier regressor such as RandomForestClassifier()
    
    '''

    print("\n")
    print("Selecting features with backward elimination: started")
    print("\n")

    # Importing the necessary libraries for Forward stepwise feature selection
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    # Let's create our feature selector
    sfs_back = SFS(wrapper_classifier,   # we can use any scikit-learn classifier or regressor.
          k_features=k_features,           # number of features we want to keep
          forward=False,
          floating=False,
          scoring = 'accuracy',
          cv = cv,                         # Cross-validation
          n_jobs=-1)                              # Run the cross-validation on all our available CPU cores.

    # Fit the model
    sfs_back.fit(X, y)

    # Print selected features
    print("\n")
    print("Features to keep in the dataframe:")
    print(sfs_back.k_feature_names_)
    print("\n")
    
    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = sfs_back.k_feature_names_)
    
    print("\n")
    print("Backward Elimination: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def exhaustive(X, y, wrapper_classifier, min_features, max_features):
    '''
    The exhaustive feature selection is a brute-force evaluation of feature subsets that evaluates model performance (such as classification accuracy) with all feature combinations. For instance, if you have 3 features, the model will be tested with feature 0 only, then feature 1 only, feature 2 only, feature 0 and 1, feature 0 and 2, feature 1 and 2, feature 0, 1 and 2. Like the other wrapper methods, the method is also computationally expensive (greedy algorithm) due to the search of all combinations. We can have different approaches to reduce this time such as reducing the search space.

    To implement it, we also can use the ExhaustiveFeatureSelector function from mlxtend.feature_selection library. As you can see in the script below, the class has min_features and max_features attributes to specify the minimum and maximum number of features we want in the combination.

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - wrapper_classifier: select a scikit-learn classifier regressor such as RandomForestClassifier()
        - min_features, max_features: specify the minimum and maximum number of features we want in the combination

    '''

    print("\n")
    print("Selecting features with exhaustive feature selection : started")
    print("\n")

    # Importing the necessary libraries for Forward stepwise feature selection
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

    # Let's create our feature selector
    efs = EFS(wrapper_classifier,         # we can use any scikit-learn classifier or regressor.
           min_features=min_features,
           max_features=max_features,
           scoring='accuracy',
           print_progress=True,
           cv=5)

    # Call the fit method on our feature selector and pass it the training set
    efs = efs.fit(X, y)
    
    # Print selected features
    print("\n")
    print("Features to keep in the dataframe:")
    print(efs.best_feature_names_)
    print("\n")
    
    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = efs.best_feature_names_)
    
    print("\n")
    print("Exhaustive Feature Selection: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def lasso(X, y, lasso_alpha):
    '''
    LASSO is a shrinkage method. It performs L1 regularization. As we can see, L1 regularization add a penalty to the cost based on the model complexity. In the equation above, instead of calculating the cost with a loss function (first term of the equation), there is an additional element (second term of the equation) called the regularization term used to penalize the model. In L1 regularization, it adds the absolute value of the magnitude of coefficient (the weights w). α is the complexity parameter which is non-negative and controls the amount of shrinkage. This is a hyper-parameter that we should tune. The larger the value, the greater the amount of shrinkage resulting in a more simplified model. If α is 0, it leads to no elimination of the parameters, increasing it leads to increase bias and decreasing it will increase the variance.

    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - lasso_alpha: α is the complexity parameter which is non-negative and controls the amount of shrinkage
    '''

    print("\n")
    print("Selecting features with LASSO: started")
    print("\n")
    
    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectFromModel
   
    # Create our feature selector
    sel_ = SelectFromModel(Lasso(alpha=lasso_alpha)) # alpha parameter is defined
    # Call the fit method on our feature selector and pass it the data
    sel_.fit(X, y)

    # Build a list of selected features and print it
    selected_feat = X.columns[(sel_.get_support())]
    print(selected_feat)

    # Print total features, selected features and features with coefficients shrank to zero
    print('total features: {}'.format((X.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = selected_feat)
    
    print("\n")
    print("Lasso DataFrame")
    print(df_data)
    
    return df_data
    
# Selecting features with regularization embedded into machine learning algorithms
def feat_reg_ml(X, y, ml_penalty):
    '''
    In embedded methods, we can select an algorithm for classification or regression and choose the penalty we want to apply.  Let’s say we want to build a model using for example a linear support vector classification algorithm (LinearSVC in scikit-learn) using a L1 penalty (Lasso for regression tasks and LinearSVC for classification).
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
        - ml_penalty: A scikit-learn classifier such as LinearSVC(C=0.05, penalty='l1', dual=False, max_iter = 5000)
    
    '''

    print("\n")
    print("Selecting features with regularization embedded into machine learning algorithms: started")
    print("\n")
    
    from sklearn.feature_selection import SelectFromModel

    # using ML algorithms with penalty
    selection = SelectFromModel(ml_penalty)
    # Call the fit method on our feature selector and pass it the data
    selection.fit(X, y)

    # see the selected features.
    selected_features = X.columns[(selection.get_support())]
    print("Selected Features: \n")
    print(selected_features)
    print("\n")

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = selected_features)
    
    print("\n")
    print("Regularization embedded into machine learning: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def embedded_linear_regression(X, y, k_features, output_folder=None):
    '''
    Here we use linear regression to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    
    print("\n")
    print("Linear Regression Features Importance: started")
    print("\n")

    # define the model
    model = LinearRegression()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.coef_
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features
    print("\n")
    print("Features Importances:")
    print("\n")
    
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Linear_Regression_Features_Importance.csv', index=False)


    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Linear Regression Feature Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Linear_Regression_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Linear Regression Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data

def embedded_logistic_regression(X, y, k_features, output_folder = None):
    '''
    Here we use logistic regression to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    
    print("\n")
    print("Logistic Regression Features Importance: started")
    print("\n")

    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.coef_[0]
    print(importance)
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Logistic_Regression_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Logistic Regression Feature Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Logistic_Regression_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Logistic Regression Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data

'''
Tree-based algorithms such as random forest, XGBoost, decision tree or extra tree are commonly used for prediction. They can also be an alternative method to select features by telling us which of them are more important, which features are the most used, in making prediction on our target variable (classification). If we take the example of random forest, a machine learning technique used to solve regression and classification, consisting of many decision trees, each tree of the random forest can calculate the importance of a feature. Random forest algorithm can do that because of its ability to increase the pureness of the leaves. In other words, when we train a tree, feature importance is determined as the decrease in node impurity weighted in a tree (higher the increment in leaves purity, the more important the feature). We call pure when the elements belong to a single class. After a normalization, the sum of the importance scores calculated is 1. The mean decrease impurity that we call Gini index (between 0 and 1), used by random forest to estimate a feature’s importance, measures the degree or probability of a variable being wrongly classified when it is randomly chosen. The index is 0 when all elements belong to a certain class, 1 when the elements are randomly distributed across various class and 0.5 the elements are equally distributed into some classes.
'''
    
def embedded_decision_tree_regressor(X, y, k_features, output_folder=None):
    '''
    Here we use decision tree regressor to select features. We select the k best features  (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    print("\n")
    print("Decision Tree Regressor Features Importance: started")
    print("\n")

    # define the model
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Decision_Tree_Regressor_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Decision Tree Regressor Features Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Decision_Tree_Regressor_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Decision Tree Regressor Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data

def embedded_decision_tree_classifier(X, y, k_features, output_folder=None):
    '''
    Here we use decision tree classifier to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    
    print("\n")
    print("Decision Tree Regressor Features Importance: started")
    print("\n")

    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Decision_Tree_Classifier_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Decision Tree Classifier Features Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Decision_Tree_Classfier_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Decision Tree Classifier Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data

def embedded_random_forest_regressor(X, y, k_features, output_folder = None):
    '''
    Here we use random forest regressor to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    
    print("\n")
    print("Random Forest Regressor Features Importance: started")
    print("\n")

    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Random_Forest_Regressor_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Random Forest Regressor Features Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Random_Forest_Regressor_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Random Forest Regressor Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def embedded_random_forest_classifier(X, y, k_features, output_folder = None):
    '''
    Here we use random forest classifier to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    print("\n")
    print("Random Forest Classifier Features Importance: started")
    print("\n")

    # define the model
    model = RandomForestClassifier()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Random_Forest_Classifier_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Random Forest Classifier Features Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Random_Forest_Classifier_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Random Forest Classifier Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def embedded_xgboost_regression(X, y, k_features, output_folder = None):
    '''
    Here we use XGboost regressor to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    
    print("\n")
    print("XGBoost Regression Features Importance: started")
    print("\n")

    # define the model
    model = GradientBoostingRegressor()
    # fit the model
    model.fit(X, y)
    
    # get importance
    importance = model.feature_importances_
    
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'XGBoost_Regression_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('XGBoost Regression Features Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'XGBoost_Regression_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("XGBoost Regression Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data

def embedded_xgboost_classification(X, y, k_features, output_folder = None):
    '''
    Here we use XGboost classifier to select features. We select the k best features (k_features)
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    
    print("\n")
    print("XGBoost Classification Features Importance: started")
    print("\n")

    # define the model
    model = GradientBoostingClassifier()
    # fit the model
    model.fit(X, y)
    
    # get importance
    importance = model.feature_importances_
    
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'XGBoost_Classification_Features_Importance.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('XGBoost Classification Features Importance')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'XGBoost_Classification_Features_Importance.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("XGBoost Classification Features Importance: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
'''
The idea of permutation feature importance was introduced by Breiman in 2001 for random forests. It measures the importance of a feature, by computing, after permuting the values of the feature, the increase in the model’s prediction error. If randomly shuffling the values of a feature increases the model error, it means that the feature is “important”. At the contrary, if shuffling the values of a feature leaves the model error unchanged, the feature is not important because the model ignored the feature for the prediction. In other words, if we destroy the information contained in a feature by randomly shuffling the feature values, the accuracy of our models should decrease. In addition, if the decrease is important, it means that the information contained in the feature is important for our predictions.

Let’s say we have trained a model and measured its quality through MSE, log-loss, etc. For each feature in the data set, we randomly shuffle the data in the feature while keeping the values of other features constant, we generate a new model based on the shuffled values and re-evaluate the quality of the new trained model, and we calculate the feature importance based on the calculation of the decrease in the quality of our new model relative to the original one. We do it for all features allowing us to rank all the features in terms of predictive usefulness.

Let’s use permutation feature importance with KNN for regression and classification.

'''
    
def embedded_permutation_regression(X, y, k_features, output_folder = None):
    '''
    Here we use permutation regressor to select features. We select the k best features k_features.
    
    Inputs:
        - X (features) DataFrame
        - y (target) DataFrame
    
    '''
    print("\n")
    print("Permutation Features Importance for Regression: started")
    print("\n")

    # define the model
    model = KNeighborsRegressor()
    # fit the model
    model.fit(X, y)
    
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean
    
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    if output_folder is not None:
        features_importance.to_csv(output_folder+'Permutation_Features_Importance_regression.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Permutation Feature Importance for Regression')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Permutation_Features_Importance_regression.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Permutation Feature Importance for Regression: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    
def embedded_permutation_classification(X, y, k_features, output_folder = None):
    print("\n")
    print("Permutation Features Importance for Classification: started")
    print("\n")

    # define the model
    model = KNeighborsClassifier()
    # fit the model
    model.fit(X, y)
    
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean
    
    # Get features name
    feature_names = [f"{i}" for i in X.columns]

    # create a data frame to visualize features importance
    features_importance = pd.DataFrame({"Features": feature_names, "Importances":importance})
    features_importance.set_index('Importances')

    # Print features importance
    print("\n")
    print("Features Importances:")
    print("\n")
    print(features_importance)
    
    features_importance.to_csv(output_folder+'Permutation_Features_Importance_Classification.csv', index=False)

    if output_folder is not None:
        # plot feature importance
        features_importance.plot(kind='bar',x='Features',y='Importances')
        pyplot.title('Permutation Features Importance for Classification')
        pyplot.tight_layout()
        pyplot.savefig(output_folder+'Permutation_Features_Importance_Classification.png')
    
    # Select the k most important features
    features_columns = []
    # Order the features importance dataframe
    df = pd.DataFrame(data = features_importance.sort_values(by='Importances', key=abs,ascending=False))
    # Put the k most important features in features_columns
    for x in range(k_features):
        features_columns = features_columns + [df.iloc[x][0]]

    # Create a new DataFrame with selected features
    df_data = pd.DataFrame(data = X, columns = features_columns)
    
    print("\n")
    print("Permutation Features Importance for Classification: DataFrame")
    print("\n")
    print(df_data)
    
    return df_data
    

