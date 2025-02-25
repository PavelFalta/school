import random

def blind_algo(tmax):
    minf = float('inf')
    for i in range(tmax):
        x = f"{random.randint(0, 999999999):b}"
        if gama(x) < minf:
            minf = gama(x)
    return minf


def gama(tn):
    return int(tn, 2)


print(blind_algo(100000))