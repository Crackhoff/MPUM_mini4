import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from common import Metric

class Hierarchical:
    def __init__(self, k, metric='euclidean', linkage='single', track_below =0):
        self.k = k
        self.labels = None
        self.metric = Metric(metric)
        self.linkage = linkage
        # print("Hierarchical clustering with", linkage, "linkage")
        self.track_below = track_below
    
    def cluster_distance(self, A, B, D):
        if self.linkage == 'single':
            min_dist = np.inf
            for a in A:
                for b in B:
                    if D[a, b] < min_dist:
                        min_dist = D[a, b]
            return min_dist
            
        elif self.linkage == 'complete':
            max_dist = -np.inf
            for a in A:
                for b in B:
                    if D[a, b] > max_dist:
                        max_dist = D[a, b]
            return max_dist
        elif self.linkage == 'average':
            for a in A:
                for b in B:
                    dist += D[a, b]
            return dist / (len(A) * len(B))
        elif self.linkage == 'centroid':
            centroid_A = np.mean(self.X[A], axis=0)
            centroid_B = np.mean(self.X[B], axis=0)
            return self.metric(centroid_A, centroid_B)
        elif self.linkage == 'ward':
            centroid = np.mean(self.X[A + B], axis=0)
            sum_a = 0
            sum_b = 0
            for a in A:
                sum_a += self.metric(self.X[a], centroid)**2
            for b in B:
                sum_b += self.metric(self.X[b], centroid)**2
            return sum_a + sum_b
        
    def combine_distances(self, dist_a, dist_b, dist_ab, size_a, size_b, size_c):
        alpha_a =0
        alpha_b = 0
        beta = 0
        gamma = 0
        if self.linkage == 'single':
            alpha_a = 0.5
            alpha_b = 0.5
            beta = 0
            gamma = -0.5
        elif self.linkage == 'complete':
            alpha_a = 0.5
            alpha_b = 0.5
            beta = 0
            gamma = 0.5
        elif self.linkage == 'average':
            alpha_a = size_a / (size_a + size_b)
            alpha_b = size_b / (size_a + size_b)
            beta = 0
            gamma = 0
        elif self.linkage == 'centroid':
            alpha_a = size_a / (size_a + size_b)
            alpha_b = size_b / (size_a + size_b)
            beta = - size_a*size_b / (size_a + size_b)**2
            gamma = 0
        elif self.linkage == 'ward':
            alpha_a = (size_a + size_c) / (size_a + size_b + size_c)
            alpha_b = (size_b + size_c) / (size_a + size_b + size_c)
            beta = - size_c / (size_a + size_b + size_c)
            gamma = 0
            
        return alpha_a * dist_a + alpha_b * dist_b + beta * dist_ab + gamma * np.abs(dist_a - dist_b)
            
        
    def fit(self, X):
        # using Lance-Williams formula
        self.X = X
        self.Xrangex = np.max(X[:, 0]) - np.min(X[:, 0])
        self.Xrangey = np.max(X[:, 1]) - np.min(X[:, 1])
        #initialize matrix
        n = X.shape[0]
        self.tracking = []
        D = cdist(X, X, metric=self.metric)
        # add diagonal
        for i in range(n):
            D[i, i] = np.inf
        #initialize clusters
        clusters = []
        for i in range(n):
            clusters.append([i])
        #initialize labels
        self.labels = np.arange(n)
        # reduce number of clusters to k
        cluster_count = n
        while cluster_count > self.k:
            
            if(cluster_count % 100 == 0):
                print(cluster_count, "clusters left")
            
            # find closest clusters
            i, j = (np.unravel_index(D.argmin(), D.shape))

            if cluster_count <= self.track_below:
                self.tracking.append({"clusters": clusters.copy(), "n_o": cluster_count})

            clusters[i] += clusters[j]
            clusters[j] = []
            self.labels = np.where(self.labels == j, i, self.labels)
            cluster_count -= 1
            # update distance matrix
            for k in range(D.shape[0]):
                # print(i, j, k, n, D.shape, len(clusters))
                if k != i and k != j:
                    D[i, k] = self.combine_distances(D[i,k], D[j,k], D[j,i], len(clusters[i]), len(clusters[j]), len(clusters[k]))
                    D[k, i] = D[i, k]
            D[i, i] = np.inf
                
            D = np.delete(D, j, axis=0)
            
            D = np.delete(D, j, axis=1)
            clusters.pop(j)
            
            for x in clusters:
                if x == []:
                    print("Empty cluster")
                    clusters.remove(x)            
        
        # print("Final clusters", clusters)
        
       
        # recolor labels
        new_labels = np.zeros(n)
        for i, cluster in enumerate(clusters):
            for j in cluster:
                new_labels[j] = i        
        
        self.labels = new_labels
        
        # print("Final labels", np.unique(self.labels))
        
        
    def animate(self):
        print("Animating last", len(self.tracking), "merges")
        fig = plt.figure()
        ax = plt.axes(xlim=self.Xrangex, ylim=self.Xrangey)   
        def update(frame):
            colors = np.zeros(self.X.shape[0])
            for i, cluster in enumerate(self.tracking[frame]["clusters"]):
                for j in cluster:
                    colors[j] = i
            ax.cla()
            ax.scatter(self.X[:, 0], self.X[:, 1], c=colors, cmap='viridis', marker='o')
            ax.set_title(f'Iteration {frame + 1}')
            ax.set_xlim(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1)
            ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)
        anim = FuncAnimation(fig, update, frames=len(self.tracking), repeat=False, interval=1000)
        anim.save('../gif/hierarchical.gif')
        plt.close()
                    
    def plot_clusters(self):
        if self.X.shape[1] == 2:
            plt.scatter(self.X[:,0], self.X[:,1], c=self.labels)
            plt.show()
        
        if self.X.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.X[:,0], self.X[:,1], self.X[:,2], c=self.labels)
            plt.show()
    
    def get_clusters(self):
        return self.labels
    
    
    
    
# class Agglomerative(Hierarchical):
#     def __init__(self, k):
#         super().__init__(k)
        
#     def fit(self, X):
#         pass
    
#     def predict(self, X):
#         pass
    
#     def score(self, X):
#         pass
    
# class Divisive(Hierarchical):
#     def __init__(self, k):
#         super().__init__(k)
        
#     def fit(self, X):
#         pass
    
#     def predict(self, X):
#         pass
    
#     def score(self, X):
#         pass