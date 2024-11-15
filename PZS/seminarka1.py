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

    # Calculate the median and MAD for initial threshold
    median = np.median(processed_signal)
    mad = np.median(np.abs(processed_signal - median))
    initial_threshold = median + initial_threshold_factor * mad

    # Apply a low-pass filter to reduce high-frequency noise
    def lowpass_filter(signal, cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, signal)

    filtered_processed_signal = lowpass_filter(processed_signal, 15, sampling_rate)

    # Initial peak detection
    peaks, properties = find_peaks(filtered_processed_signal, height=initial_threshold, distance=min_distance)

    # Dynamic thresholding and searchback
    SPKI = 0
    NPKI = 0
    Threshold_I1 = initial_threshold
    Threshold_I2 = 0.5 * Threshold_I1
    RR_intervals = []
    r_peaks = []

    for i, peak in enumerate(peaks):
        if filtered_processed_signal[peak] > Threshold_I1:
            SPKI = 0.125 * filtered_processed_signal[peak] + 0.875 * SPKI
            r_peaks.append(peak)
            if len(r_peaks) > 1:
                RR_intervals.append(r_peaks[-1] - r_peaks[-2])
        else:
            NPKI = 0.125 * filtered_processed_signal[peak] + 0.875 * NPKI

        Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
        Threshold_I2 = 0.5 * Threshold_I1

        # Searchback mechanism
        if len(RR_intervals) > 0 and (peak - r_peaks[-1]) > 1.66 * np.mean(RR_intervals):
            searchback_window = filtered_processed_signal[r_peaks[-1]:peak]
            searchback_peaks, _ = find_peaks(searchback_window, height=Threshold_I2)
            if len(searchback_peaks) > 0:
                max_peak = searchback_peaks[np.argmax(filtered_processed_signal[searchback_peaks])]
                r_peaks.append(max_peak)
                SPKI = 0.25 * filtered_processed_signal[max_peak] + 0.75 * SPKI
                Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                Threshold_I2 = 0.5 * Threshold_I1

    # plt.plot(filtered_processed_signal)
    # plt.plot(r_peaks, filtered_processed_signal[r_peaks], 'x')
    # plt.show()

    return np.array(r_peaks)

# Example usage:
# ecg_signal = ...  # Load your ECG signal here
# sampling_rate = 360  # Example sampling rate
# processed_signal = refined_pan_tompkins_ecg_processing(ecg_signal, sampling_rate)
# r_peaks = detect_r_peaks_with_refined_threshold(processed_signal, sampling_rate)
# print(f"BPM: {len(r_peaks) / (len(ecg_signal) / sampling_rate) * 60}")

ecg_signal = load_data(data_path)
segments = np.array_split(ecg_signal, len(ecg_signal) // (hz * seconds))
bpm_list = []

old = 0
old_segment = None
old_r_peaks = None
old_ref_ecg = None

for segment in segments:
    ref_ecg = refined_pan_tompkins_ecg_processing(segment, hz)
    r_peaks = detect_r_peaks_with_refined_threshold(ref_ecg, hz)
    bpm = len(r_peaks) / seconds * 60
    bpm_list.append(bpm)
    if old != 0 and (bpm < 0.7 * old or bpm > 1.3 * old):
        print(f"Suspicious BPM change from {old} to {bpm}")
        # fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        # axs[0, 0].plot(old_segment)
        # axs[0, 0].plot(old_r_peaks, old_segment[old_r_peaks], 'x')
        # axs[0, 0].set_title(f'Previous Segment (Raw) BPM: {old}')
        # axs[1, 0].plot(old_ref_ecg)
        # axs[1, 0].plot(old_r_peaks, old_ref_ecg[old_r_peaks], 'x')
        # axs[1, 0].set_title('Previous Segment (Processed)')
        # axs[0, 1].plot(segment)
        # axs[0, 1].plot(r_peaks, segment[r_peaks], 'x')
        # axs[0, 1].set_title(f'Current Segment (Raw) BPM: {bpm}')
        # axs[1, 1].plot(ref_ecg)
        # axs[1, 1].plot(r_peaks, ref_ecg[r_peaks], 'x')
        # axs[1, 1].set_title('Current Segment (Processed)')
        # plt.show()
    old = bpm
    old_segment = segment
    old_r_peaks = r_peaks
    old_ref_ecg = ref_ecg

# print(f"BPM list: {bpm_list}")
print(f"Mean BPM: {np.mean(bpm_list)}")

plt.figure(figsize=(12, 6))
plt.plot(bpm_list)
plt.xlabel('Segment')
plt.ylabel('BPM')
plt.title('BPM over time')
plt.show()

# Apply polynomial regression to the BPM list
degrees = range(1, 50)  # Extend the range to allow for more degrees
best_degree = 1
best_r2 = -np.inf
X = np.arange(len(bpm_list)).reshape(-1, 1)
y = np.array(bpm_list)
no_improvement_count = 0

for degree in degrees:
    print(degree)
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_poly_pred = model.predict(X_poly)
    r2 = r2_score(y, y_poly_pred)
    
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree
        no_improvement_count = 0  # Reset the counter if improvement is found
    else:
        no_improvement_count += 1
    
    if no_improvement_count >= 10:
        break

# Fit the best polynomial model
poly_features = PolynomialFeatures(degree=best_degree)
X_poly = poly_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)

# Plot the BPM list and the polynomial regression model
plt.figure(figsize=(12, 6))
plt.plot(bpm_list, label='BPM')
plt.plot(y_poly_pred, label=f'Polynomial Regression (degree={best_degree})')
plt.xlabel('Segment')
plt.ylabel('BPM')
plt.title('BPM over time with Polynomial Regression')
plt.legend()
plt.show()