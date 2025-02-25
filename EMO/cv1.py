import random
import numpy as np

def kvadraticka_funkce(x):
    return 21 * x ** 2 + 95 * x - 12 # s minusem

def gama(s, k):
    bin_len_m = k
    cisla = [s[i:i+bin_len_m] for i in range(0, len(s), bin_len_m)]
    cisla = [int(i, 2) for i in cisla]
    return cisla

def minf(i_max):
    mn = float("inf")

    for i in range(i_max):
        n = 1
        k = 10

        cisla = "".join([str(random.randint(0, 1)) for _ in range(n*k)])

        gm = gama(cisla, 10)[0]

        y = kvadraticka_funkce(gm)
        if y < mn:
            mn = y
    print(mn)

minf(1000)