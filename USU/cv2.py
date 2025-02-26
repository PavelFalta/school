# k-means from sratch
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def k_means(data, clusters):
    centroids = deque(data[np.random.choice(range(len(data)), clusters)])
    labels = np.zeros(len(data))

    curr_cent = centroids.popleft()

    for i in range(100):
        # use numpy to search for the closest point to current centroid
        closest = np.argmin(np.linalg.norm(data - curr_cent, axis=1))
        