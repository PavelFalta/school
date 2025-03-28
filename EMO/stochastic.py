import random
import math

def random_search(f, tmax, k, n):
    f_min = math.inf
    alpha_min = None
    # Hlavní smyčka pro hledání náhodného řešení
    for time in range(tmax):

        alpha = random_alpha(size=k*n)
        x = gamma(alpha, k, n)
        f_x = f(gamma(alpha, k, n))

        # Kontrola, zda je nové řešení lepší než dosavadní nejlepší
        if f_x < f_min:
            f_min =  f_x
            alpha_min = alpha
        print(f"t: {time}, alpha: {alpha}, x: {x}, f(x): {f_x}, f_min: {f_min}")
    return (alpha_min, f_min)

# Funkce pro generování náhodného binárního řetězce
def random_alpha(size):
    return "".join([str(random.randint(0, 1)) for i in range(size)])

# Funkce pro převod binárního řetězce na seznam čísel
def gamma(alfa, k, n):
    x = [int(alfa[i:i+k], 2) for i in range(0, k*n, k)]
    return x

def main():
    # Spuštění náhodného hledání s danou funkcí a parametry
    random_search(f = lambda x: -x[0]**2 + 6, tmax = 30, k = 4, n = 1)

if __name__ == "__main__":
    main()
