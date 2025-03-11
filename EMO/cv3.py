from collections import deque
from random import randint
import random

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

def negace(gen, i):
    gen = list(gen)
    gen[i] = 1 - int(gen[i])
    return gen

def U(alpha, S):
    return [negace(alpha, i) for i in S]

def tabu_search(f, tmax, k, n, maxlen):
    f_min = float("inf")
    alpha_min = None
    tabu = deque(maxlen=maxlen)

    for t in range(tmax):
        alpha = random_alpha(size=k*n)

        S = [i for i in range(k*n) if not i in tabu]
        _U = U(alpha, S)
        
        alpha_min = min(_U, key=lambda x: f(gamma(x, k, n)))

        x = gamma(alpha_min, k, n)

        current_f = f(x)

        if current_f < f_min:
            f_min = current_f

        best_i = S[_U.index(alpha_min)]
        tabu.append(best_i)

        print(f"t: {t}, alpha: {alpha_min}, x: {x}, f(x): {current_f}, f_min: {f_min}, tabu: {tabu}")
    return (alpha_min, f_min)

tabu_search(f = lambda x: sum((xi - 5)**2 for xi in x), tmax = 100, k = 4, n = 5, maxlen=10)