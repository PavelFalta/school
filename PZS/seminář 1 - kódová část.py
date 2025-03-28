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

    def load_data(self):
        # načtení dat z dané cesty
        data = np.fromfile(self.data_path, dtype=np.int16)
        return data

    def pan_tompkins(self, i, ecg_signal, lowcut=1, highcut=30, filter_order=2, window_duration=0.12):
        # aplikace bandpass filtru
        filtered_signal = self._bandpass_filter(ecg_signal, lowcut, highcut, self.hz, filter_order)
        # diferenciace signálu
        differentiated_signal = np.diff(filtered_signal, prepend=filtered_signal[0])
        # kvadratizace signálu
        squared_signal = differentiated_signal ** 2
        # aplikace pohyblivého průměru
        window_size = int(window_duration * self.hz)
        mwi_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

        # kontrola stability signálu
        if i > 20 and np.std(mwi_signal) < 0.01 * np.mean(self.prev_std):
            self.bpm_tag.append(i)
        else:
            self.prev_std.append(np.std(mwi_signal))

        return mwi_signal

    def detect_r_peaks(self, processed_signal, initial_threshold_factor=0.5):
        # nastavení minimální vzdálenosti mezi vrcholy
        min_distance = int(self.hz * 60 / 250)
        # výpočet počátečního prahu
        initial_threshold = self._calculate_initial_threshold(processed_signal, initial_threshold_factor)
        # aplikace lowpass filtru
        filtered_processed_signal = self._lowpass_filter(processed_signal, 15, self.hz)
        # detekce vrcholů
        peaks, properties = find_peaks(filtered_processed_signal, height=initial_threshold, distance=min_distance)

        # dynamické prahování
        r_peaks = self._dynamic_thresholding(peaks, filtered_processed_signal, initial_threshold)
        # filtrování vrcholů
        r_peaks = self._filter_peaks(r_peaks, filtered_processed_signal)

        return np.array(r_peaks)

    def polymer_regression(self, x, y):
        best_r2 = -np.inf
        best_y_pred = None
        best_degree = 1
        no_improvement_count = 0
        degree = 1

        # hledání nejlepšího stupně polynomu
        while no_improvement_count < 20:
            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x.reshape(-1, 1))
            model = LinearRegression()
            model.fit(x_poly, y)
            y_pred = model.predict(x_poly)
            r2 = r2_score(y, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_y_pred = y_pred
                best_degree = degree
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            degree += 1

        print(f"nejlepší stupeň: {best_degree} s r2: {best_r2}")
        return best_y_pred

    def _bandpass_filter(self, signal, lowcut, highcut, fs, order):
        # aplikace bandpass filtru
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def _lowpass_filter(self, signal, cutoff, fs, order=2):
        # aplikace lowpass filtru
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, signal)

    def _calculate_initial_threshold(self, signal, factor):
        # výpočet počátečního prahu
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return median + factor * mad

    def _dynamic_thresholding(self, peaks, signal, initial_threshold):
        signal_peak = 0
        noise_peak = 0
        threshold_high = initial_threshold
        rr_intervals = []
        r_peaks = []

        # dynamické prahování vrcholů
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

        # filtrování vrcholů
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

        return res
    
if __name__ == "__main__":
    data_path = "/home/pavel/py/school/PZS/data/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/122001/122001_ECG.dat"
    hz = 1000
    seconds = 10 

    detector = ECGPeakDetector(data_path, hz, seconds)
    ecg_signal = detector.load_data()

    print("ecg data nactena.")
    
    while True:
        start = random.randint(0, len(ecg_signal) - hz * seconds)
        segment = ecg_signal[start:start + hz * seconds]
        print(f"zpracovani segmentu od {start} do {start + hz * seconds}")

        processed_signal = detector.pan_tompkins(start, segment)
        print("signal zpracovan pomoci pan-tompkins algoritmu.")

        r_peaks = detector.detect_r_peaks(processed_signal)

        # normalizujeme signal pro lepsi vizualizaci
        processed_signal = processed_signal * (np.max(segment) / np.max(processed_signal))
        bpm = 60 * len(r_peaks) / seconds

        print(f"detekovano {len(r_peaks)} r-vrcholu v segmentu, BPM {bpm}")

        plt.figure(figsize=(12, 6))
        plt.plot(np.linspace(0, seconds, len(segment)), segment, label="Původní ECG", color='deepskyblue')
        plt.plot(np.linspace(0, seconds, len(processed_signal)), processed_signal, label="Předzpracovaný signál", color='red')
        plt.plot(np.array(r_peaks) / hz, processed_signal[r_peaks], "x", label="Detekované R-vrcholy", color='black')
        plt.legend()
        plt.title(f"Vizualizace Detekce R-vrcholu v Segmentu, BPM: {bpm}")
        plt.xlabel("Sekundy")
        plt.ylabel("Amplituda")
        plt.show()
