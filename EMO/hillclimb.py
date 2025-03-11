import random
import math

# Funkce pro hill climbing algoritmus
def hill_climb(f, tmax, k, n, c0, pmut):
    f_min = math.inf  # Inicializace minimální hodnoty funkce
    alpha_min = None  # Inicializace nejlepšího řešení
    alpha = random_alpha(size=k*n)  # Generování náhodného alfa
    for time in range(tmax):
        U = [mutation(alpha, pmut) for i in range(c0)]  # Generování nových řešení mutací
        alpha = min(U, key=lambda x: f(gamma(x, k, n)))  # Výběr nejlepšího řešení
        x = gamma(alpha, k, n)  # Převod alfa na x
        f_x = f(x)  # Výpočet hodnoty funkce
        if f_x < f_min:
            f_min =  f_x  # Aktualizace minimální hodnoty funkce
            alpha_min = alpha  # Aktualizace nejlepšího řešení
        print(f"t: {time}, alpha: {alpha}, x: {x}, f(x): {f_x}, f_min: {f_min}")
    return (alpha_min, f_min)  # Návrat nejlepšího řešení a jeho hodnoty

# Funkce pro mutaci
def mutation(alpha, pmut):
    return ''.join(
        str(1 - int(bit)) if random.random() < pmut else bit
        for bit in alpha
    )

# Funkce pro generování náhodného alfa
def random_alpha(size):
    return ''.join(random.choice('01') for _ in range(size))

# Funkce pro převod alfa na x
def gamma(alpha, k, n):
    return [gray_dec(alpha[i*k:(i+1)*k]) for i in range(n)]

# Funkce pro převod Grayova kódu na desítkovou soustavu
def gray_dec(alpha):
    
    binary = int(alpha[0])
    result = binary
    for bit in alpha[1:]:
        binary ^= int(bit)
        result = (result << 1) | binary
    return result

# Hlavní funkce
def main():
    print(hill_climb(f = lambda x: -x[0]**2 + 6, tmax = 30, k = 4, n = 1, c0=10, pmut=0.1))

if __name__ == "__main__":
    main()
