
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
    
    
    prev_labels = deque([None], maxlen=1)
    prev_labels.append(labels.copy())
    
    while not converged:
        
        current_labels = labels.copy()

        
        for i, point in enumerate(data):
            distances = deque([distance(point, centroid) for centroid in centroids])
            labels[i] = np.argmin(distances)
        
        
        for j in range(clusters):
            cluster_points = deque(data[labels == j])
            if len(cluster_points) > 0:
                centroids[j] = np.mean(list(cluster_points), axis=0)
        
        
        prev_labels.append(labels.copy())
        if np.array_equal(prev_labels[0], current_labels):
            converged = True
    
    
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.title(f'K-means with {clusters} clusters')
    plt.show()
    
    return centroids, labels

    
k_means(X, 1)