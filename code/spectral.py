import numpy as np
from common import Metric
from kmeans import KMeans
from matplotlib import pyplot as plt
from hierarchical import Hierarchical

# Spectral Clustering -> KMeans
class Spectral:
    def __init__(self, k, metric='euclidean', kmeans=None):
        self.k = k
        self.labels = None
        self.metric = Metric(metric)
        if kmeans == None:
            self.kmeans = k
        else:
            self.kmeans = kmeans
        
    def create_laplacian(self, X):
        n = X.shape[0]
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                W[i, j] = np.exp(-self.metric(X[i], X[j])**2)
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        # normalized laplacian
        D_inv_sqrt = np.diag(1 / np.sqrt(np.sum(W, axis=1)))
        L = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
        
        return L
    
    def get_eigenvectors(self, L):
        eigvals, eigvecs = np.linalg.eigh(L)
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        return eigvecs
    
    def split_on_k_eigens(self, eigvecs):
        U = eigvecs[:, 1:self.k+1]
        # print(U.shape, self.kmeans)
        kmeans = KMeans(self.kmeans)
        kmeans.fit(U)
        self.labels = kmeans.labels
        self.centroids = kmeans.centroids
        # print(self.centroids)
        return self.labels        
        
    def fit(self, X):
        self.X = X
        L = self.create_laplacian(X)
        eigvecs = self.get_eigenvectors(L)
        self.split_on_k_eigens(eigvecs)
        
    def plot_clusters(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        # adding centroids does not work at all
        # plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()
        
    def get_clusters(self):
        return self.labels
        
    def rate_clustering(self):
        return self.kmeans.rate_clustering()