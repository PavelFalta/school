import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks


def bandpass_filter(signal, lowcut, highcut, fs, order):
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def pan_tomkins(ecg_signal, lowcut=1, highcut=30, filter_order=2, window_duration=0.12):
    filtered_signal = bandpass_filter(ecg_signal, lowcut, highcut, 360, filter_order)
    
    differentiated_signal = np.diff(filtered_signal, prepend=filtered_signal[0])
    
    squared_signal = differentiated_signal ** 2

    window_size = int(window_duration * 360)
    mwi_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

    return mwi_signal

def detect_r_peaks(processed_signal, fs, initial_threshold_factor=0.5):
    min_distance = int(fs * 60 / 250)
    initial_threshold = calculate_initial_threshold(processed_signal, initial_threshold_factor)
    filtered_processed_signal = lowpass_filter(processed_signal, 15, fs)
    peaks, properties = find_peaks(filtered_processed_signal, height=initial_threshold, distance=min_distance)
    r_peaks = dynamic_thresholding(peaks, filtered_processed_signal, initial_threshold)
    r_peaks = filter_peaks(r_peaks, filtered_processed_signal)
    return np.array(r_peaks)

def filter_peaks(r_peaks, signal):
    res = []
    if r_peaks.size > 0:
        max_peak = np.argmax(signal[r_peaks])
        res.append(r_peaks[max_peak])
        r_peaks = np.delete(r_peaks, max_peak)

    for i, peak in enumerate(r_peaks):
        distance = peak - r_peaks[i-1] if i > 0 else np.mean(np.diff(r_peaks))
        mean_amplitude = np.mean(signal[r_peaks])
        peak_amplitude = signal[peak]

        if distance < 0.7 * np.mean(np.diff(r_peaks)):
            if peak_amplitude < 0.5 * mean_amplitude:
                continue

        res.append(peak)

    return np.array(res)

def lowpass_filter(signal, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def calculate_initial_threshold(signal, factor):
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    return median + factor * mad

def dynamic_thresholding(peaks, signal, initial_threshold):
    signal_peak = 0
    noise_peak = 0
    threshold_high = initial_threshold
    rr_intervals = []
    r_peaks = []

    for peak in peaks:
        if signal[peak] > threshold_high:
            signal_peak = 0.125 * signal[peak] + 0.875 * signal_peak
            r_peaks.append(peak)
            if len(r_peaks) > 1:
                rr_intervals.append(r_peaks[-1] - r_peaks[-2])
        else:
            noise_peak = 0.125 * signal[peak] + 0.875 * noise_peak

        threshold_high = noise_peak + 0.05 * (signal_peak - noise_peak)

    return np.array(r_peaks)


EKG = electrocardiogram()

Fs = 360
T = 1/Fs

N = 5*3840

for i in range(5):
    curr_EKG = EKG[N*i:N*(i+1)]
    t = np.arange(0, len(curr_EKG)/Fs, T)

    mwi_signal = pan_tomkins(curr_EKG)



    r_peaks = detect_r_peaks(mwi_signal, Fs)


    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, curr_EKG, label='EKG signál')
    plt.plot(t[r_peaks], curr_EKG[r_peaks], 'ro', label='R-vlny')
    plt.legend()
    plt.title('EKG signál s R-vlnami')

    plt.subplot(2, 1, 2)
    plt.plot(t, mwi_signal, label='Filtrovaný EKG signál')
    plt.plot(t[r_peaks], mwi_signal[r_peaks], 'ro', label='R-vlny')
    plt.legend()
    plt.title('Filtrovaný EKG signál s R-vlnami')

    plt.tight_layout()
    plt.show()

    spectrum = np.fft.fft(curr_EKG)
    log_spectrum = np.log(np.abs(spectrum))
    cepstrum = np.fft.ifft(log_spectrum).real

    plt.figure(figsize=(12, 6))
    plt.plot(cepstrum)
    plt.title('Cepstrum of EKG signal')
    plt.xlabel('Quefrency')
    plt.ylabel('Amplitude')
    plt.show()