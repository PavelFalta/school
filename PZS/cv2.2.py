import numpy as np
import scipy as sp
from scipy import signal, integrate
import matplotlib.pyplot as plt


t = np.linspace(0, 10, 5000)

funkce = np.sin(2*np.pi*0.5*t)**np.sin(2*t)

t_kernelu = np.linspace(0, 1, 500)

alpha = 1
a = 2

kernel = alpha*np.exp(-a*t_kernelu)

plt.plot(kernel)
plt.show()
funkce = np.where(t >= 1.0, 0, funkce)

plt.plot(funkce)


conv = np.convolve(funkce, kernel, mode = 'full')/300

plt.plot(conv)

plt.show()

def convolve(kernel, x):
    res = []

    #kernel = np.flip(kernel)
    kernel = np.tile(kernel, 10)

    for i, g in enumerate(x):

        a = kernel[i:i+2]
        b = x[i:i+2]

        ia = integrate.simpson(y=a)
        ib = integrate.simpson(y=b)

        res.append(max(ib-ia, 0))

    return res




r = convolve(kernel, funkce)

plt.plot(funkce)

plt.plot(r)
plt.show()