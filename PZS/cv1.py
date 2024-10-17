import numpy as np
import matplotlib.pyplot as plt

A0 = 1
f = 2
M = 3
fi = 22

def sin(t, A0, f):
    W = 2*np.pi*f
    return A0*np.sin(W*t)

def mod_sin(M, t, fi):

    return M*np.sin(t + fi)

def ampl_mod(A0, M, t, f, fi):
    return A0 + mod_sin(M, t, fi)*sin(t, A0, f)



t = np.linspace(0,10/2,5000)

plt.plot(sin(t, A0, f), label = "nosna_vlna")
plt.plot(mod_sin(M, t, fi), label = "mod_vlna")
plt.plot(ampl_mod(A0, M, t, f, fi), label = "ampl_vlna")
plt.legend()
plt.show()