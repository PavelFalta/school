from collections import deque
from random import randint


def negace(gen, i):
    gen[i] = 1 - gen[i]
    return gen

def U(alpha, S, T):
    return [S(alpha) for i in S if i not in T]

gen = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
gen = negace(gen, 3)
print(gen)