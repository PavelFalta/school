
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

X, y = make_blobs(n_samples=10000, centers=11, n_features=2, random_state=0)

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def k_means(data, clusters):

    centroids = data[np.random.choice(range(len(data)), clusters, replace=False)]

    labels = np.zeros(len(data))

    old_labels = np.eye(len(data), dtype=int)

    while not np.all(labels == old_labels):
        
        old_labels = labels.copy()

        for i, point in enumerate(data):
            distances = [distance(point, centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        for j in range(clusters):
            points_in_cluster = [data[i] for i in range(len(data)) if labels[i] == j]
            if points_in_cluster:  
                centroids[j] = np.mean(points_in_cluster, axis=0)
    
    return centroids, labels

    
centroids, labels = k_means(X, 11)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)
plt.show()