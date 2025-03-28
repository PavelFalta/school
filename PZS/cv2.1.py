import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

A0 = 1        
nosna_frekvence = 5       
modulacni_frekvence = 10       
frekvenci_zdvih = 2*np.pi*5
E = 0        


def nosna_vlna(t, A0, nosna_frekvence, E):
    w_c = 2 * np.pi * nosna_frekvence
    return A0 * np.sin(w_c * t + E)

def mod_vlna(t, modulacni_frekvence, frekvenci_zdvih):
    return frekvenci_zdvih * np.cos(2 * np.pi * modulacni_frekvence * t)

def frekvence_modulace(t, A0, nosna_frekvence, frekvenci_zdvih, modulacni_frekvence, E):
    omega_t = 2 * np.pi * nosna_frekvence + mod_vlna(t, modulacni_frekvence, frekvenci_zdvih)
    return A0 * np.sin(omega_t * t + E)


t = np.linspace(0, 10, 10000)


nosna = nosna_vlna(t, A0, nosna_frekvence, E)
modulacni = mod_vlna(t, modulacni_frekvence, frekvenci_zdvih)
fm_signal = frekvence_modulace(t, A0, nosna_frekvence, frekvenci_zdvih, modulacni_frekvence, E)


plt.plot(t, nosna, label="Nosná vlna")
plt.plot(t, modulacni, label="Modulační vlna")
plt.plot(t, fm_signal, label="Frekvenčně modulovaný signál")
plt.title("Frekvenční modulace")
plt.xlabel("Čas [s]")
plt.ylabel("Amplituda")
plt.legend()
plt.grid(True)
plt.show()