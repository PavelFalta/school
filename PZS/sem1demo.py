import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import random
import time
import matplotlib.pyplot as plt

class ECGPeakDetector:
    def __init__(self, data_path, hz, seconds):
        self.data_path = data_path
        self.hz = hz
        self.seconds = seconds
        self.segment_data = []
        self.bpm_list = []
        self.bpm_reg = None
        self.prev_std = []
        self.bpm_tag = []
        self.last_peak_num = 0

    def load_data(self):
        data = np.fromfile(self.data_path, dtype=np.int16)
        return data

    def pan_tompkins(self, i, ecg_signal, lowcut=1, highcut=30, filter_order=2, window_duration=0.12):
        filtered_signal = self._bandpass_filter(ecg_signal, lowcut, highcut, self.hz, filter_order)
        differentiated_signal = np.diff(filtered_signal, prepend=filtered_signal[0])
        squared_signal = differentiated_signal ** 2
        window_size = int(window_duration * self.hz)
        mwi_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

        if i > 20 and np.std(mwi_signal) < 0.01 * np.mean(self.prev_std):
            self.bpm_tag.append(i)
        else:
            self.prev_std.append(np.std(mwi_signal))

        return mwi_signal

    def detect_r_peaks(self, processed_signal, initial_threshold_factor=0.5):
        min_distance = int(self.hz * 60 / 250)
        initial_threshold = self._calculate_initial_threshold(processed_signal, initial_threshold_factor)
        filtered_processed_signal = self._lowpass_filter(processed_signal, 15, self.hz)
        peaks, properties = find_peaks(filtered_processed_signal, height=initial_threshold, distance=min_distance)

        r_peaks = self._dynamic_thresholding(peaks, filtered_processed_signal, initial_threshold)
        r_peaks = self._filter_peaks(r_peaks, filtered_processed_signal)

        return np.array(r_peaks)

    def polymer_regression(self, x, y):
        best_r2 = -np.inf
        best_y_pred = None
        best_degree = 1
        no_improvement_count = 0
        degree = 1

        while no_improvement_count < 20:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(x.reshape(-1, 1))
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            r2 = r2_score(y, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_y_pred = y_pred
                best_degree = degree
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            degree += 1

        print(f"Best degree: {best_degree} with R2: {best_r2}")
        return best_y_pred

    def _bandpass_filter(self, signal, lowcut, highcut, fs, order):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def _lowpass_filter(self, signal, cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, signal)

    def _calculate_initial_threshold(self, signal, factor):
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return median + factor * mad

    def _dynamic_thresholding(self, peaks, signal, initial_threshold):
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

            threshold_high = noise_peak + 0.25 * (signal_peak - noise_peak)

        return r_peaks

    def _filter_peaks(self, r_peaks, signal):
        res = []
        if r_peaks:
            max_peak = np.argmax(signal[r_peaks])
            res.append(r_peaks.pop(max_peak))

        for i, peak in enumerate(r_peaks):
            distance = peak - r_peaks[i-1] if i > 0 else np.mean(np.diff(r_peaks))
            mean_amplitude = np.mean(signal[r_peaks])
            peak_amplitude = signal[peak]

            if distance < 0.7 * np.mean(np.diff(r_peaks)):
                if peak_amplitude < 0.5 * mean_amplitude:
                    continue
            elif peak_amplitude < 0.4 * mean_amplitude:
                continue

            res.append(peak)

        if self.last_peak_num:
            while len(res) > self.last_peak_num + 2:
                min_peak_index = np.argmin(signal[res])
                res.pop(min_peak_index)
            while len(res) < self.last_peak_num - 2 and len(r_peaks) > 0:
                max_peak_index = np.argmax(signal[r_peaks])
                res.append(r_peaks[max_peak_index])
                r_peaks = np.delete(r_peaks, max_peak_index)
        self.last_peak_num = len(res)

        return res
    
if __name__ == "__main__":
    data_path = "/home/pavel/py/school/PZS/data/mit-bih-normal-sinus-rhythm-database-1.0.0/18184.dat"
    hz = 128  
    seconds = 10 

    detector = ECGPeakDetector(data_path, hz, seconds)
    ecg_signal = detector.load_data()

    print("ECG data loaded.")
    
    while True:
        start = random.randint(0, len(ecg_signal) - hz * seconds)
        segment = ecg_signal[start:start + hz * seconds]
        print(f"Processing segment from {start} to {start + hz * seconds}")

        processed_signal = detector.pan_tompkins(start, segment)
        print("Signal processed using Pan-Tompkins algorithm.")

        r_peaks = detector.detect_r_peaks(processed_signal)

        # Normalize the processed signal to have the same amplitude as the segment signal
        processed_signal = processed_signal * (np.max(segment) / np.max(processed_signal))


        print(f"Detected {len(r_peaks)} R-peaks in the segment.")

        plt.figure(figsize=(12, 6))
        plt.plot(segment, label="Original ECG Signal", color='lightblue')
        plt.plot(processed_signal, label="Processed Signal", color='red')
        plt.plot(r_peaks, processed_signal[r_peaks], "x", label="Detected R-peaks")
        plt.legend()
        plt.title(f"ECG Segment from {start} to {start + hz * seconds} and Detected R-peaks")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()