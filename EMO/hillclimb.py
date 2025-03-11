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
def mutation(alfa, pmut):
    child = [] 
    for i in range(len(alfa)):
        child.append(str(1 + int(alfa[i]) * -1) if random.random() < pmut else alfa[i])
    return "".join(child)

# Funkce pro generování náhodného alfa
def random_alpha(size):
    return "".join([str(random.randint(0, 1)) for i in range(size)])

# Funkce pro převod alfa na x
def gamma(alfa, k, n):
    return [gray2dec(alfa[i*k:(i+1)*k]) for i in range(n)]

# Funkce pro převod Grayova kódu na desítkovou soustavu
def gray2dec(alfa):
    dual = [alfa[0]]
    for nextbinary in alfa[1:]:
        dual.append(str(int(dual[-1]) ^ int(nextbinary)))
    return int("".join(dual), 2)

# Hlavní funkce
def main():
    print(hill_climb(f = lambda x: -x[0]**2 + 6, tmax = 30, k = 4, n = 1, c0=10, pmut=0.1))

if __name__ == "__main__":
    main()
