def secti(a, b):
    return a + b

def odecti(a, b):
    return a - b


# Testovaci scenar - napr testovani toho, jestli se admin muze prihlasit do systemu
# TS = N x TP

class TesterKalkulacky: # testovaci trida
    
    def otestuj_secti(): # testovaci funkce/metoda
        assert secti(1, 2) == 3 # testovací příklad
        assert secti(-2, 3) == 1 # testovací příklad 2

    def otestuj_odecti():
        assert odecti(1, 2) == -1
        assert odecti(-2, 3) == -5