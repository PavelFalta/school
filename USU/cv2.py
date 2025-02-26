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
    centroids = deque(data[np.random.choice(range(len(data)), clusters)])
    print(centroids)
    labels = np.zeros(len(data))

    curr_cent = centroids.popleft()

    # plt.plot(curr_cent, 'ro')
    # plt.scatter(data[:, 0], data[:, 1], c=labels)
    # plt.show()
    for i in range(len(data)):
        distances = [distance(data[i], c) for c in centroids]
        labels[i] = np.argmin(distances)
    
k_means(X, 5)