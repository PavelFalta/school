from collections import deque
from random import randint


def negace(gen, i):
    gen[i] = 1 - gen[i]
    return gen

def U(alpha, S):
    return [negace(i, randint(0, len(i))) for i in S]

gen = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
gen = negace(gen, 3)
print(gen)