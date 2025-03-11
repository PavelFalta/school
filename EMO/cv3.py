def negace(gen, i):
    gen[i] = 1 - gen[i]
    return gen

gen = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
gen = negace(gen, 3)
print(gen)