import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def aprox(n, t):
    result = np.zeros_like(t)
    for i in range(1, n + 1):
        result += (2 / i) * np.cos(i * np.pi) * np.sin(i * t)
    return -result

def f(x):
    y = x
    return y

start = 1
end = 1000
iter_step = 200

t = np.linspace(-np.pi, np.pi, 400)
# t = np.linspace(-10, 10, 400)

plt.plot(t, f(t), label='f')

for n in range(start, end + 1, iter_step):
    plt.plot(t, aprox(n, t), label=f'n={n}')

plt.legend()
plt.xlabel('t')
plt.ylabel('f(t)')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
