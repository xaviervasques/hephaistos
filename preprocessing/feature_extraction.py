
#!/usr/bin/python3
# feature_extraction.py
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


# Importing all packages required for imputation
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA

"""

The advantage of future extraction is that we will select and/or combine variables into features and reduce the amount of redundant data. This process will considerably reduce the
amount of data that needs to be processed. Of course, the idea is not to compromise with accuracy. The idea is to reduce data and at the same time keep accuracy, or even more, improve
accuracy. To summarize, features extraction allows to avoid the risk of overfitting, speed up in training, can improve accuracy and a better data visualization and explainability of
our models. Feature extraction is used in image processing to detect features in an image or video. It can be a shape or motion. Another practical case is natural language processing
with what is called Bag-of-Words. The idea is to extract the words (features) in a document or whatever and classify them by frequency of use. Autoencoders, unsupervised learning of
efficient data coding, is also a good application where feature extraction will help identify important features for coding.

In this section, we will discuss some of the linear and nonlinear dimensionality reduction techniques which are widely used in a variety of applications such as Principal Component
Analysis (PCA), Independent Component Analysis (ICA), Linear Discriminant Analysis (LDA) or Locally Linear Embedding (LLE). Once we understood how these methods work, we can explore
many more methods such as  Canonical Correlation Analysis (CCA), Singular Value Decomposition (SVD), CUR Matrix Decomposition, Compact Matrix Decomposition (CMD), Non Negative Matrix
Factorization (NMF), Kernel PCA, Multidimensional Scaling (MDS), Isomap, Laplacian Eigen map, Local Tangent Space Alignment (LTSA) or Fast map.

"""

def pca(df, number_components = None):

    """

    Data transformation using unsupervised learning is a common method to better visualize, compress and extract the most informative data. Principal component analysis (PCA) is a
    well-known algorithm and widely used to transform correlated features into features that are not statistically correlated. This transformation is followed by the selection of a
    subgroup of new features classified by order of importance which can best summarize the original data distribution. It allows to reduce data’s original dimensions. The basic
    idea is that PCA will study p variables measured on n individuals. When n and p are large, the aim is to reduce the number of features of a data set, while preserving as much
    information as possible.

    """
    
    # define PCA. n_components: Number of principal components we want to keep and defined in inputs.py
    pca = PCA(n_components=number_components)
    # transform data
    principal_components = pca.fit_transform(df.copy())
    # To create principal component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['PCA_%i'%x]
    df_pca = pd.DataFrame(data = principal_components, columns = component_columns)
    print('\n')
    print('PCA: Explained Variance Ratio')
    print(pca.explained_variance_ratio_)
    print('\n')
    return df_pca

def ica(df, number_components = None):

    """

    Independent component analysis (ICA) separates a multivariate signal into additive subcomponents. we need first to make two assumptions to apply ICA. The independent components
    that are hidden and that we try to isolate must be statistically independent and non-Gaussian. The difference between PCA and ICA is that the first one compresses information
    and the second one separates information. Both require autoscaling but ICA can benefit from first applying PCA. Here is without applying PCA.

    """

    ica = FastICA(n_components=number_components)
    principal_components = ica.fit_transform(df.copy())
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['ICA_%i'%x]
    df_ica = pd.DataFrame(data = principal_components, columns = component_columns)
    return df_ica
    
def icawithpca(df, number_components):

    """

    Independent component analysis (ICA) separates a multivariate signal into additive subcomponents. we need first to make two assumptions to apply ICA. The independent components
    that are hidden and that we try to isolate must be statistically independent and non-Gaussian. The difference between PCA and ICA is that the first one compresses information
    and the second one separates information. Both require autoscaling but ICA can benefit from first applying PCA. Here, we apply PCA.

    """

    # Apply PCA first
    pca = PCA(n_components=number_components)
    principal_components = pca.fit_transform(df.copy())
    # Apply ICA on PCA data
    ica = FastICA(n_components=number_components)
    I = ica.fit_transform(principal_components)
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['ICA_with_PCA_%i'%x]
    df_icawithpca = pd.DataFrame(data = I, columns = component_columns)
    return df_icawithpca
        

def lda_extraction(X_train,y_train,X_test,y_test, number_components = None):
    """
    
    LDA is a supervised machine learning technique used to classify data. It is also used as a dimensionality reduction technique to project the features from a higher dimension space
    into a lower dimension space with good class-separability avoiding overfitting and reduce computational costs. PCA and LDA are comparable in the sense that they are linear
    transformations. LDA will maximize the distance between the mean of each class and minimize the spreading within the class itself (minimizing variation between each category).
    
    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        
    """
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=number_components)
    
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['LDA_%i'%x]
            
    df_train = pd.DataFrame(data = lda.fit(X_train,y_train).transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = lda.fit(X_test,y_test).transform(X_test), columns = component_columns)
    
    return df_train, df_test
    
def random_projection(X_train,y_train,X_test,y_test, number_components = None):
    """
    
    Sparse random projection is a dimensionality reduction technique which is an alternative to dense random projection matrix that guarantees similar embedding quality while being
    much more memory efficient and allowing faster computation of the projected data.

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
    
    """
    from sklearn.random_projection import SparseRandomProjection
    # Load and apply Random projection
    ran = SparseRandomProjection(n_components=number_components, random_state=42)
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['Random_Projection_%i'%x]
        
    df_train = pd.DataFrame(data = ran.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = ran.fit_transform(X_test), columns = component_columns)
    return df_train, df_test
    
def truncatedSVD(X_train,y_train,X_test,y_test, number_components = None):
    """
    Truncated SVD is a dimensionality reduction technique performing linear dimensionality reduction by means of truncated singular value decomposition (SVD).

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)

    """
    
    from sklearn.decomposition import TruncatedSVD
    # Load and apply Truncated SVD
    trun = TruncatedSVD(n_components=number_components)
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['TruncatedSVD_%i'%x]
        
    df_train = pd.DataFrame(data = trun.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = trun.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def isomap(X_train,y_train,X_test,y_test, number_components = None, n_neighbors = None):
    """
    Isomap Embedding is a non-linear dimensionality reduction approach through Isometric Mapping

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        - n_neighbors as defined in inputs.py (Number of neighbors to take into account for Manifold Learning techniques)

    """
    from sklearn.manifold import Isomap
    # Load and apply Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=number_components)
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['Isomap_%i'%x]
        
    df_train = pd.DataFrame(data = isomap.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = isomap.fit_transform(X_test), columns = component_columns)
    return df_train, df_test
    
def standard_lle(X_train,y_train,X_test,y_test, number_components = None, n_neighbors = None):
    """
    
    LLE is an unsupervised learning algorithm that computes low-dimensional, neighborhood preserving embeddings of high-dimensional inputs. LLE is based on Manifold Learning which is
    a class of unsupervised estimators aiming to describe datasets as low-dimensional manifolds embedded in high-dimensional spaces.
    
    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        - n_neighbors as defined in inputs.py (Number of neighbors to take into account for Manifold Learning techniques)
        
    """
    
    from sklearn.manifold import LocallyLinearEmbedding
    # Load and apply Standard LLE
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=number_components, method="standard")
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['standard_lle_%i'%x]
        
    df_train = pd.DataFrame(data = lle.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = lle.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def modified_lle(X_train,y_train,X_test,y_test, number_components = None, n_neighbors = None):
    """
    
    MLLE addresses regularization problem by using multiple weight vectors in each neighbourhood.
    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        - n_neighbors as defined in inputs.py (Number of neighbors to take into account for Manifold Learning techniques)
        
    """
    
    from sklearn.manifold import LocallyLinearEmbedding
    # Load and apply Modified LLE
    mlle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=number_components, method="modified")
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['modified_lle_%i'%x]
        
    df_train = pd.DataFrame(data = mlle.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = mlle.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def hessian_lle(X_train,y_train,X_test,y_test, number_components = None, n_neighbors = None):
    """
    
    HLLE achieves linear embedding by minimizing the Hessian functional on the manifold (hessian-based quadratic form at each neighbourhood used to recover the locally linear
    structure). The scaling is not optimal with the increase of data size but tends to give higher quality results compared to standard LLE.
    
    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        - n_neighbors as defined in inputs.py (Number of neighbors to take into account for Manifold Learning techniques)
    
    """
    
    from sklearn.manifold import LocallyLinearEmbedding
    # Load and apply Hessian LLE
    hlle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=number_components, method="hessian")
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['hessian_lle_%i'%x]
        
    df_train = pd.DataFrame(data = hlle.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = hlle.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def ltsa_lle(X_train,y_train,X_test,y_test,number_components = None, n_neighbors = None):
    """
    
    In LTSA, PCA is applied on the neighbours to construct a locally linear patch considered as an approximation of the tangent space at the point. A coordinate representation of the
    neighbours is provided by the tangent place and gives a low-dimensional representation of the patch.

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        - n_neighbors as defined in inputs.py (Number of neighbors to take into account for Manifold Learning techniques)
    
    """
    
    from sklearn.manifold import LocallyLinearEmbedding
    # Load and apply LTSA LLE
    ltsalle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=number_components, method="ltsa")
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['ltsa_lle_%i'%x]
        
    df_train = pd.DataFrame(data = ltsalle.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = ltsalle.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def mds(X_train,y_train,X_test,y_test, number_components = None):
    """
    
    Multidimensional Scaling is a non-linear technique for embedding data in a lower-dimensional space. It maps points residing in a higher-dimensional space to a lower-dimensional
    space while preserving the distances between those points as much as possible.

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
    
    """
    
    from sklearn.manifold import MDS
    # Load and apply MDS
    mds = MDS(n_components=number_components, n_init=1, max_iter=120, n_jobs=2)
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['mds_%i'%x]
        
    df_train = pd.DataFrame(data = mds.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = mds.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def spectral(X_train,y_train,X_test,y_test, number_components = None):
    """
    
    Spectral Embedding is a technique used for non-linear dimensionality reduction (Laplacian Eigenmaps).

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
    
    """
    
    from sklearn.manifold import SpectralEmbedding
    # Load and apply Spectral
    spectral = SpectralEmbedding(n_components=number_components, random_state=0, eigen_solver="arpack")
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['spectral_%i'%x]
        
    df_train = pd.DataFrame(data = spectral.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = spectral.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def tsne(X_train,y_train,X_test,y_test,number_components = None):
    """
    
    t-SNE is an unsupervised and non-linear technique well suited to visualize high-dimensional data by converting similarities between data points to joint probabilities. t-SNE
    minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE models the dataset with a
    dimension-agnostic probability distribution allowing to find a lower-dimensional approximation with a closely matching distribution
    
    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
        - Parameters we may want to tune: n_iter, n_iter_without_progress, ...
    """
    
    from sklearn.manifold import TSNE
    # Load and apply t-SNE
    tsne = TSNE(
        n_components=number_components,
        init="pca",
        learning_rate="auto",
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0)
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['tsne_%i'%x]
        
    df_train = pd.DataFrame(data = tsne.fit_transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = tsne.fit_transform(X_test), columns = component_columns)
    return df_train, df_test

def nca(X_train,y_train,X_test,y_test, number_components = None):
    """
    Neighbourhood components analysis is a supervised learning method for classifying multivariate data into distinct classes according to a given distance metric over the data.

    Inputs:
        - splitted training and testing data from original dataframe as defined in inputs.py
        - number_components as defined in inputs.py (Number of principal components we want to keep)
    
    """
    
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    # Load and apply NCA
    nca = NeighborhoodComponentsAnalysis(n_components=number_components, init="pca", random_state=0)
    # To create component columns
    component_columns = []
    for x in (n+1 for n in range(number_components)):
        component_columns = component_columns + ['nca_%i'%x]
        
    df_train = pd.DataFrame(data = nca.fit(X_train,y_train).transform(X_train), columns = component_columns)
    df_test = pd.DataFrame(data = nca.fit(X_test,y_test).transform(X_test), columns = component_columns)
    return df_train, df_test
