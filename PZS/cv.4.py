import numpy as np

import matplotlib.pyplot as plt

frequency = 5
period = 1 / frequency
amplitude = 1
noise_amplitude = 15
sampling_rate = 1000
duration = 200 * np.pi

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

signal = amplitude * np.sin(2 * np.pi * frequency * t)

def increasing_noise(amp, t):
    return amp * np.random.normal(size=t.shape) * (t / t.max())

noisy_signal = signal + increasing_noise(noise_amplitude, t)

plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Periodic Signal')
plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.7)
plt.legend()
plt.show()

num_periods = int(duration * frequency)
samples_per_period = int(sampling_rate / frequency)
noisy_signal_matrix = noisy_signal[:num_periods * samples_per_period].reshape((num_periods, samples_per_period))

plt.plot(noisy_signal_matrix[0])
plt.show()

noisy_signal_matrix = noisy_signal_matrix * np.linspace(1, 0, num_periods).reshape(-1, 1)

mean_signal = np.mean(noisy_signal_matrix, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(mean_signal)
plt.show()