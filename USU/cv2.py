# k-means from sratch
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt

X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()