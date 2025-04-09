from cvxopt import matrix, solvers
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10, 100)
y = x + np.random.normal(size=100)

plt.scatter(x, y)
plt.show()
