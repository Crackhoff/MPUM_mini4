import numpy as np
import matplotlib.pyplot as plt
class Metric:
    def __init__(self, metric):
        self.metric = metric
        
    def __call__(self, a, b):
        # metrics work not only in 2D, but in any dimension
        if self.metric == 'euclidean':
            return np.linalg.norm(a - b)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        elif self.metric == 'adam':
            return np.sum(a != b)
        elif self.metric == 'l_inf':
            return np.max(np.abs(a - b))
        
def compare_clustering(cl_A, cl_B, full_info=False):
    # compare two clusterings - find the best match
    n = np.unique(cl_A)
    m = np.unique(cl_B)
    clusters_A = len(n)
    clusters_B = len(m)
    # create contingency table
    cont = np.zeros((clusters_A, clusters_B))
    for i in range(clusters_A):
        for j in range(clusters_B):
            for k in range(len(cl_A)):
                if cl_A[k] == n[i] and cl_B[k] == m[j]:
                    cont[i, j] += 1
    # print(cont)
    
    fittness = np.zeros(clusters_A)
    matchings = np.zeros(clusters_A)
    
    for i in range(clusters_A):
        fittness[i] = np.max(cont[i])
        matchings[i] = np.argmax(cont[i]).astype(int)
        
    if full_info:
        return np.sum(fittness) / len(cl_A), matchings        
    return np.sum(fittness) / len(cl_A)

def TSNE(data, n_components=2):
    """
    Reduce the dimension of the data using TSNE
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.manifold import TSNE
    return TSNE(n_components=n_components).fit_transform(data)

def PCA(data, n_components=2):
    """
    Reduce the dimension of the data using PCA
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components).fit_transform(data)

def MDS(data, n_components=2):
    """
    Reduce the dimension of the data using MDS
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.manifold import MDS
    return MDS(n_components=n_components).fit_transform(data)

def Isomap(data, n_components=2):
    """
    Reduce the dimension of the data using Isomap
    :param data: pandas dataframe
    :param n_components: number of components to reduce to
    :return: numpy array
    """
    from sklearn.manifold import Isomap
    return Isomap(n_components=n_components).fit_transform(data)