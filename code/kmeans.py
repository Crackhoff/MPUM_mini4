import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from common import Metric

class KMeans:
    def __init__(self, k, metric='euclidean', max_iter=1000):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.metric = Metric(metric)
        self.history = []
        
    def initialize_centroids(self, X):
        # create k random centroids
        self.centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            self.centroids[i] = X[np.random.randint(0, X.shape[0])]
        return self.centroids
            
    def assign_points(self, X):
        # assign each point to the nearest centroid
        for i in range(X.shape[0]):
            min_dist = np.inf
            for j in range(self.k):
                dist = self.metric(X[i], self.centroids[j])**2 # squared metric
                # dist = self.metric(X[i], self.centroids[j]) # non-squared metric
                if dist < min_dist: # takes care of ties
                    min_dist = dist
                    self.labels[i] = j
        
    def fit(self, X):
        self.X = X
        self.X_rangex = np.max(X[:, 0]) - np.min(X[:, 0])
        self.X_rangey = np.max(X[:, 1]) - np.min(X[:, 1])
        self.labels = np.zeros(X.shape[0])
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            old_centroids = self.centroids
            self.assign_points(X)
            self.centroids = np.zeros((self.k, X.shape[1]))
            for i in range(self.k):
                if np.sum(self.labels == i) == 0:
                    self.centroids[i] = X[np.random.randint(0, X.shape[0])]
                else:
                    self.centroids[i] = np.mean(X[self.labels == i], axis=0)

            self.history.append((self.labels.copy(), self.centroids.copy()))
            if np.all(old_centroids == self.centroids):
                break
    
    def plot_clusters(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
        plt.show()
        
    def plot_history(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(0,31), ylim=(0, 31))

        def update(frame):
            ax.cla()
            labels, centroids = self.history[frame]
            ax.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis', marker='o')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
            ax.set_title(f'Iteration {frame + 1}')
            ax.set_xlim(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1)
            ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)

        anim = FuncAnimation(fig, update, frames=len(self.history), repeat=False)
        anim.save('../gif/kmeans.gif')
        
        plt.close()
        
    def rate_clustering(self):
        # calculate sum of squared distances from centroids
        sse = 0
        for i in range(self.X.shape[0]):
            # print(self.X[i], self.labels[i].astype(int))
            sse += self.metric(self.X[i], self.centroids[self.labels[i].astype(int)])**2
        return sse
    
    def get_clusters(self):
        return self.labels
    
    
class KMeansPP(KMeans):
    def __init__(self, k, metric='euclidean', max_iter=2000):
        super().__init__(k, metric, max_iter)
    
    def initialize_centroids(self, X):
        # first centroid is random
        self.centroids = np.zeros((self.k, X.shape[1]))
        self.centroids[0] = X[np.random.randint(0, X.shape[0])]
        
        # initialize list of distances - no need to remove chosen points - prob will be 0
        
        for i in range(1, self.k):
            dist = np.zeros(X.shape[0])
            for j in range(X.shape[0]):
                dist[j] = np.min([self.metric(X[j], self.centroids[l])**2 for l in range(i)])
            # square
            dist = dist**2
            dist = dist / np.sum(dist)
            self.centroids[i] = X[np.random.choice(X.shape[0], p=dist)]
            
        return self.centroids
    
    def plot_history(self):
        fig = plt.figure()
        ax = plt.axes(xlim=self.X_rangex, ylim=self.X_rangey)

        def update(frame):
            ax.cla()
            labels, centroids = self.history[frame]
            ax.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis', marker='o')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
            ax.set_title(f'Iteration {frame + 1}')
            ax.set_xlim(self.X[:, 0].min() - 1, self.X[:, 0].max() + 1)
            ax.set_ylim(self.X[:, 1].min() - 1, self.X[:, 1].max() + 1)

        anim = FuncAnimation(fig, update, frames=len(self.history), repeat=False)
        anim.save('../gif/kmeanspp.gif')
        
        plt.close()
        
def find_k(X, k, iter =1, metric='euclidean', max_iter=1000):
    # search for the best k and plot the results
    scores = {}
    
    for i in range(2, k+1, iter):
        avg = 0
        # for _ in range(10):  
        kmeans = KMeans(i, metric, max_iter)
        kmeans.fit(X)
        # avg+=kmeans.rate_clustering()
        # scores.append(avg/10)
        scores[i] = kmeans.rate_clustering()
        
    plt.plot(range(2, k+1, iter), scores.values())
    # add points to the plot
    for i in range(2, k+1, iter):
        plt.scatter(i, scores[i], c='red')
    plt.show()
    
    # find the angle of the elbow
    angles = {}
    for i in range(2, k-1, iter):
        x1 = i
        x2 = i+1
        y1 = scores[i]
        y2 = scores[i+1]
        m = (y2-y1)/(x2-x1)
        angle = np.arctan(m)
        angles[i] = angle
    
    return np.argmax(angles) + 2