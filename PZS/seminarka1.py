import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, detrend
import scipy.signal as signal

data_path = "PZS/data/100001_ECG.dat"
hz = 1000
seconds = 10

def load_data(data_path):
    data = np.fromfile(data_path, dtype=np.int16)
    return data


def refined_pan_tompkins_ecg_processing(ecg_signal, sampling_rate, lowcut=1, highcut=30, filter_order=2, window_duration=0.12):

    def bandpass_filter(signal, lowcut, highcut, fs, order):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    filtered_signal = bandpass_filter(ecg_signal, lowcut, highcut, sampling_rate, filter_order)
    differentiated_signal = np.diff(filtered_signal, prepend=filtered_signal[0])
    squared_signal = differentiated_signal ** 2
    window_size = int(window_duration * sampling_rate)
    mwi_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

    return mwi_signal

def detect_r_peaks_with_refined_threshold(processed_signal, sampling_rate, initial_threshold_factor=0.5):

    min_distance = int(sampling_rate * 60 / 250)  # Minimum distance corresponding to 250 BPM
    initial_threshold = np.mean(processed_signal) + initial_threshold_factor * np.std(processed_signal)
    peaks, properties = find_peaks(processed_signal, height=initial_threshold, distance=min_distance)
    plt.plot(processed_signal)
    plt.plot(peaks, processed_signal[peaks], 'x')
    plt.show()

    return peaks

ecg_signal = load_data(data_path)
segments = np.array_split(ecg_signal, len(ecg_signal) // (hz * seconds))
bpm_list = []

for segment in segments:
    ref_ecg = refined_pan_tompkins_ecg_processing(segment, hz)
    r_peaks = detect_r_peaks_with_refined_threshold(ref_ecg, hz)
    bpm = len(r_peaks) / seconds * 60
    bpm_list.append(bpm)
    if bpm < 50:
        print(f"Segment with BPM < 50: {bpm}")
        plt.plot(segment)
        plt.plot(ref_ecg)
        plt.plot(r_peaks, segment[r_peaks], 'x')
        plt.show()

# print(f"BPM list: {bpm_list}")

plt.figure(figsize=(12, 6))
plt.plot(bpm_list)
plt.xlabel('Segment')
plt.ylabel('BPM')
plt.title('BPM over time')
plt.show()