import numpy as np
import matplotlib.pyplot as plt

M = np.ones((4, 4), dtype=complex)

for i in range(1, 4):
    for j in range(1, 4):
        M[i, j] = np.exp(-2j * np.pi * i * j / 4)

print(M)
V = [[8],
     [4],
     [8],
     [0]]

res = M @ V
print(np.abs(res))
plt.plot(np.abs(res))
plt.show()