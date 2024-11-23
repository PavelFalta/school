from time import perf_counter
from functools import *

# cache = {5:8, 9:2}
#fibonacci 5
# 5 in cache:
#else:
#

@cache
def fibonnaci(n):
    if n == 0:
        return 0
    
    if n == 1:
        return 1
    
    return fibonnaci(n-1) + fibonnaci(n-2)

start = perf_counter()
print(fibonnaci(32))

print(perf_counter()-start)


def brian_kerningan(n):
    print(bin(n))

    count = 0

    while n:
        n = n & (n - 1)
        count += 1

    return count

print(brian_kerningan(255))