from collections import deque
from random import randint
import random

def negace(gen, i):
    gen = list(gen)
    gen[i] = 1 - int(gen[i])
    return gen

def U(alpha, S, tabu):
    return [negace(alpha, i) for i in S if i not in tabu]

def random_alpha(size):
    return ''.join(random.choice('01') for _ in range(size))

def gamma(alpha, k, n):
    return [gray_dec(alpha[i*k:(i+1)*k]) for i in range(n)]

def gray_dec(alpha):
    
    binary = int(alpha[0])
    result = binary
    for bit in alpha[1:]:
        binary ^= int(bit)
        result = (result << 1) | binary
    return result

def tabu_search(f, tmax, k, n, c0, pmut, S, maxlen):
    f_min = float("inf")
    alpha_min = None
    tabu = deque(maxlen=maxlen)

    for t in range(tmax):
        alpha = random_alpha(size=k*n)
        print(alpha)
        
        #print(f"t: {time}, alpha: {alpha}, x: {x}, f(x): {f_x}, f_min: {f_min}")
    return (alpha_min, f_min)

tabu_search(f = lambda x: -x[0]**2 + 6, tmax = 30, k = 4, n = 5, c0=10, pmut=0.1, S=[0, 1, 2, 3], maxlen=10)