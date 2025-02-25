import random
from matplotlib import pyplot as plt
import numpy as np

def kvadraticka_funkce(a, b, c, x):
    return a * x ** 2 + b * x - c # s minusem

def gama(n, k):
    cisla = [random.randint(1, 100) for _ in range(n)]
    m = max(cisla)
    bin_len_m = len(bin(m)) - 2
    assert bin_len_m <= k, "Nelze zakodovat"
    bin_len_m = max(bin_len_m, k)
    cisla = [bin(i)[2:].zfill(bin_len_m) for i in cisla]
    print("".join(cisla))


t = np.arange(-10, 10, 0.1)
y = kvadraticka_funkce(1, 1, 1, t)
plt.plot(t, y)
plt.show()

