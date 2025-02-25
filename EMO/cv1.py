import random
import numpy as np

def kvadraticka_funkce(a, b, c, x):
    return a * x ** 2 + b * x - c # s minusem

def binary_kod(cisla, k):
    m = max(cisla)
    bin_len_m = len(bin(m)) - 2

    assert bin_len_m <= k, "Nelze zakodovat"

    bin_len_m = max(bin_len_m, k)
    cisla = [bin(i)[2:].zfill(bin_len_m) for i in cisla]
    return "".join(cisla)

def gama(s, k):
    bin_len_m = k
    cisla = [s[i:i+bin_len_m] for i in range(0, len(s), bin_len_m)]
    cisla = [int(i, 2) for i in cisla]
    return cisla

def minf(i_max):
    min = float("inf")

    for i in range(i_max):

        cisla = [random.randint(1, 100) for _ in range(4)]

        x = binary_kod(cisla, 10)
        der_x = gama(x, 10)

        a = der_x[0]
        b = der_x[1]
        c = der_x[2]
        x = der_x[3]

        y = kvadraticka_funkce(a, b, c, x)
        if y < min:
            min = y
    print(min)

minf(100)