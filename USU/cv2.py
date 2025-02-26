
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def k_means(data, clusters):

    centroids = data[np.random.choice(len(data), clusters, replace=False)]
    
    labels = np.zeros(len(data))
    converged = False

    history_centroids = []
    history_centroids.append(centroids)
    
    while not converged:
        
        old_labels = labels.copy()

        for i, point in enumerate(data):
            distances = [distance(point, centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        
        for j in range(clusters):
            if len(data[labels == j]) > 0:  
                centroids[j] = np.mean(data[labels == j], axis=0)
        
        
        if np.array_equal(labels, old_labels):
            converged = True
    
    
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
        plt.show()

        history_centroids.append(centroids)
    
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.plot(history_centroids)
    plt.show()
    
    return centroids, labels

    
k_means(X, 2)