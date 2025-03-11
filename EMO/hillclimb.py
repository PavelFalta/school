

import random
import math

def hill_climb(f, tmax, k, n, c0, pmut):
    f_min = math.inf
    alpha_min = None
    alpha = random_alpha(size=k*n)
    for time in range(tmax):
        U = [mutation(alpha, pmut) for i in range(c0)]
        alpha = min(U, key=lambda x: f(gamma(x, k, n)))
        x = gamma(alpha, k, n)
        f_x = f(x)
        if f_x < f_min:
            f_min =  f_x
            alpha_min = alpha
        print(f"t: {time}, alpha: {alpha}, x: {x}, f(x): {f_x}, f_min: {f_min}")
    return (alpha_min, f_min)

def mutation(alfa, pmut):
    child = "" 
    for i in range(len(alfa)):
        child += str(1 + int(alfa[i]) * -1) if random.random() < pmut else alfa[i]
    return child

def random_alpha(size):
    return "".join([str(random.randint(0, 1)) for i in range(size)])

def gamma(alfa, k, n):
    return [gray2dec(alfa[i*k:(i+1)*k]) for i in range(n)]

def gray2dec(alfa):
    dual = [alfa[0]]
    for nextbinary in alfa[1:]:
        dual.append(str(int(dual[-1]) ^ int(nextbinary)))
    return int("".join(dual), 2)

def main():
    print(hill_climb(f = lambda x: -x[0]**2 + 6, tmax = 30, k = 4, n = 1, c0=10, pmut=0.1))

if __name__ == "__main__":
    main()

