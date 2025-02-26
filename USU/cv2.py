# k-means from sratch
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def k_means(data, clusters):

    # Initialize centroids with random points from data
    indices = np.random.choice(len(data), clusters, replace=False)
    centroids = deque([data[i] for i in indices])
    
    labels = np.zeros(len(data))

    curr_cent = centroids.popleft()

    smallest = distance(data[0], curr_cent)

    data = deque(data)

    while data:
        i = data.popleft()

        dist = distance(i, curr_cent)

        if dist < smallest:
            smallest = dist
            labels[i] = curr_cent
    
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

    
k_means(X, 5)