import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import find_peaks



def s(t, f):
    return A0 * np.cos(2 * np.pi * f*t)

def noise(fnoise, t):
    return A0 * np.cos(2 * np.pi * fnoise * t)

def error(t, f, fnoise):
    nt = np.random.uniform(-5*A0, 5*A0, size=t.shape)
    return s(t, f) + noise(fnoise, t) + nt

t = np.linspace(0, 1, 1000)
A0 = 5
f = 10
fnoise = 100
y = error(t, f, fnoise)

plt.plot(t, y)
plt.plot(t, s(t, f))
plt.show()

Y = fft(y)
p = find_peaks(np.abs(Y), height=2000)

plt.plot(p[0], np.abs(Y[p[0]]), 'ro')
plt.plot(np.abs(Y))
plt.show()

Y_filtered = np.zeros_like(Y)
Y_filtered[p[0]] = Y[p[0]]
y_filtered = ifft(Y_filtered)

plt.plot(t, y_filtered.real)
plt.title('Reconstructed Signal from Peaks')
plt.show()