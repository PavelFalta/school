import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import scipy.signal as signal
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from tkinter import ttk
import os
import json
import threading
import pandas as pd
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial import ConvexHull

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
    

class ECGClusterer:
    def __init__(self, data_path, hz, seconds):
        self.data_path = data_path
        self.hz = hz
        self.seconds = seconds

    def load_data(self):
        return np.fromfile(self.data_path, dtype=np.int16)

    def preprocess_data(self, data):
        data = (data - np.mean(data)) / np.std(data)
        window_size = self.hz * self.seconds
        segments = [data[i:i + window_size] for i in range(0, len(data), window_size)]
        features = []

        f_nyquist = self.hz / 2
        vlf_band = (0.003 * f_nyquist, 0.04 * f_nyquist)
        lf_band = (0.04 * f_nyquist, 0.15 * f_nyquist)
        hf_band = (0.15 * f_nyquist, 0.4 * f_nyquist)

        for segment in segments:
            freqs, psd = signal.welch(segment, fs=self.hz, nperseg=1024)
            total_power = np.sum(psd)
            if total_power == 0:
                continue

            vlf_power = np.sum(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
            lf_power = np.sum(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
            hf_power = np.sum(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

            vlf_ratio = vlf_power / total_power if total_power > 0 else 0
            lf_ratio = lf_power / total_power if total_power > 0 else 0
            hf_ratio = hf_power / total_power if total_power > 0 else 0
            lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0
            psd_norm = psd / total_power if total_power != 0 else np.zeros_like(psd)
            spectral_entropy = entropy(psd_norm)
            mean_freq = np.sum(freqs * psd) / total_power if total_power != 0 else 0
            median_freq = freqs[np.argsort(psd)[len(psd) // 2]]
            peak_freq = freqs[np.argmax(psd)]
            bandwidth = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * psd) / total_power) if total_power != 0 else 0

            mean_val = np.mean(segment)
            std_val = np.std(segment)
            max_val = np.max(segment)
            min_val = np.min(segment)
            range_val = max_val - min_val
            rms_val = np.sqrt(np.mean(segment**2))
            skewness = pd.Series(segment).skew()
            kurtosis = pd.Series(segment).kurtosis()

            features.append([
                total_power, vlf_power, lf_power, hf_power, 
                vlf_ratio, lf_ratio, lf_hf_ratio, 
                spectral_entropy, mean_freq, 
                std_val, 
                max_val, min_val, range_val, rms_val, 
                skewness
            ])
        
        return np.array(features)

    def kmeans_clustering(self, features, n_clusters):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=0.95)
        reduced_features = pca.fit_transform(scaled_features)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(reduced_features)
        return kmeans.labels_
    
class ECGClusterViewer:
    def __init__(self, root, data, features, labels, hz, seconds):
        self.root = root
        self.data = data
        self.features = features
        self.labels = labels
        self.hz = hz
        self.seconds = seconds
        self.scatter = None
        self.current_index = None
        self.palette = ColorPalette()

        self._setup_ui()
        self._bind_events()
        self.plot_clusters()

    def _setup_ui(self):
        self.root.title("ECG Cluster Viewer")
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.root, bg=self.palette.current_palette['bg'])
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        self.root.configure(bg=self.palette.current_palette['bg'])
        self.button_frame.configure(bg=self.palette.current_palette['bg'])

    def _bind_events(self):
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def plot_clusters(self):
        reduced_features = self._reduce_features()
        self.scatter = self.axs[1].scatter(reduced_features[:, 0], reduced_features[:, 1], c=self.labels, cmap='viridis')
        self.fig.colorbar(self.scatter, ax=self.axs[1])
        self.axs[1].set_title('ECG Clusters')
        self.axs[1].set_xlabel('Principal Component 1')
        self.axs[1].set_ylabel('Principal Component 2')
        self._highlight_good_cluster(reduced_features)
        self.canvas.draw()

    def _reduce_features(self):
        pca = PCA(n_components=2)
        return pca.fit_transform(self.features)

    def _highlight_good_cluster(self, reduced_features):
        cluster_centers, cluster_sizes, middle_point = self._calculate_cluster_properties(reduced_features)
        good_cluster_label = self._determine_good_cluster(cluster_centers, cluster_sizes, middle_point)
        self._draw_cluster_boundaries(reduced_features, good_cluster_label)

    def _calculate_cluster_properties(self, reduced_features):
        cluster_centers = np.array([reduced_features[self.labels == label].mean(axis=0) for label in np.unique(self.labels)])
        cluster_sizes = np.array([np.sum(self.labels == label) for label in np.unique(self.labels)])
        middle_point = np.mean(reduced_features, axis=0)
        return cluster_centers, cluster_sizes, middle_point

    def _determine_good_cluster(self, cluster_centers, cluster_sizes, middle_point):
        distances_to_middle = np.linalg.norm(cluster_centers - middle_point, axis=1)
        normalized_sizes = cluster_sizes / np.max(cluster_sizes)
        normalized_distances = distances_to_middle / np.max(distances_to_middle)
        scores = normalized_sizes - normalized_distances
        good_cluster_index = np.argmax(scores)
        return np.unique(self.labels)[good_cluster_index]

    def _draw_cluster_boundaries(self, reduced_features, good_cluster_label):
        good_cluster_points = reduced_features[self.labels == good_cluster_label]
        hull = ConvexHull(good_cluster_points)
        centroid = np.mean(good_cluster_points, axis=0)
        distances_from_centroid = np.linalg.norm(good_cluster_points - centroid, axis=1)
        max_distance = np.percentile(distances_from_centroid, 95)
        filtered_points = good_cluster_points[distances_from_centroid < max_distance]
        moved_points = centroid + 1 * (filtered_points - centroid)
        if len(moved_points) >= 3:
            moved_hull = ConvexHull(moved_points)
            for i, simplex in enumerate(moved_hull.simplices):
                self.axs[1].plot(moved_points[simplex, 0], moved_points[simplex, 1], 'r--', lw=2, label='Predicted Good Signal' if not i else None)
        self.axs[1].legend()

    def on_click(self, event):
        if event.inaxes == self.axs[1]:
            closest_index = self._find_closest_point(event)
            self.plot_detail(closest_index)

    def _find_closest_point(self, event):
        x, y = event.xdata, event.ydata
        reduced_features = self._reduce_features()
        distances = np.sqrt((reduced_features[:, 0] - x) ** 2 + (reduced_features[:, 1] - y) ** 2)
        return np.argmin(distances)

    def plot_detail(self, index):
        self.current_index = index
        segment_start = index * self.hz * self.seconds
        segment_end = segment_start + self.hz * self.seconds
        segment = self.data[segment_start:segment_end]
        time_axis = np.linspace(0, self.seconds, len(segment))
        group_color = self.scatter.cmap(self.labels[index] / max(self.labels))
        self.axs[0].clear()
        self.axs[0].plot(time_axis, segment, color=group_color)
        self.axs[0].set_title(f'Segment {index}')
        self.axs[0].set_xlabel('Time (s)')
        self.axs[0].set_ylabel('Amplitude')
        self._highlight_selected_point()
        self.canvas.draw()

    def _highlight_selected_point(self):
        reduced_features = self._reduce_features()
        if hasattr(self, 'highlighted_point'):
            self.highlighted_point.remove()
        if self.current_index is not None:
            self.highlighted_point = self.axs[1].scatter(reduced_features[self.current_index, 0], reduced_features[self.current_index, 1], s=100, edgecolor='red', facecolor='none', linewidth=2)
        self.canvas.draw_idle()

    def on_closing(self):
        self.root.destroy()

class ColorPalette:
    day_palette = {
        'bg': 'white',
        'fg': 'black',
        'plot_bg': 'white',
        'plot_fg': 'black',
        'line_color': 'blue',
        'peak_color': 'red',
        'bpm_color': '#39FF14',
        'button_bg': 'white',
        'button_fg': 'black',
        'title_color': 'black',
        'span_color': 'red',
        'hover_line_color': 'gray',
        'hover_text_color': 'gray',
        'highlight_color': 'red',
        'highlight_alpha': 0.4
    }

    night_palette = {
        'bg': '#1E1E1E',
        'fg': '#C0C0C0',
        'plot_bg': '#121212',
        'plot_fg': '#D3D3D3',
        'line_color': '#9370DB',
        'peak_color': '#FF6347',
        'bpm_color': '#39FF14',
        'button_bg': '#2E2E2E',
        'button_fg': '#C0C0C0',
        'title_color': '#D3D3D3',
        'span_color': '#FF4500',
        'hover_line_color': '#808080',
        'hover_text_color': '#808080',
        'highlight_color': '#FF4500',
        'highlight_alpha': 0.4
    }

    def __init__(self):
        self.is_night_mode = False
        self.current_palette = self.day_palette

    def toggle_night_mode(self):
        self.is_night_mode = not self.is_night_mode
        self.current_palette = self.night_palette if self.is_night_mode else self.day_palette

class ECGSegmentViewer:
    def __init__(self, root, segment_data, bpm_list, bpm_reg, bpm_tag, seconds, config_manager, file_entry):
        self.root = root
        self.segment_data = segment_data
        self.bpm_list = bpm_list
        self.bpm_reg = bpm_reg
        self.bpm_tag = bpm_tag
        self.seconds = seconds
        self.config_manager = config_manager
        self.file_entry = file_entry
        self.current_segment_index = 0
        self.palette = ColorPalette()

        self._setup_ui()
        self._bind_events()
        self.plot_segment(self.current_segment_index)
        self.default_xlim = self.axs[0, 1].get_xlim()

    def _setup_ui(self):
        self.root.title(f"ECGViewer - {os.path.basename(self.file_entry)}")
        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1.5]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self._create_buttons()

    def _create_buttons(self):
        self.button_frame = tk.Frame(self.root, bg=self.palette.current_palette['bg'])
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_button = self._create_button(self.button_frame, "Reset Zoom", self.reset_zoom)
        self.night_mode_button = self._create_button(self.button_frame, "Toggle Night Mode", self.toggle_night_mode)
        self.night_mode_button.pack(side=tk.RIGHT, padx=10, pady=5)
        self.segment_entry = self._create_entry(self.button_frame)
        self.jump_button = self._create_button(self.button_frame, "Jump to Segment", self.jump_to_segment)
        self.hint_button = self._create_button(self.button_frame, "?", self.show_hint)

    def _create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, command=command, bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'])
        button.pack(side=tk.LEFT, padx=10, pady=5)
        return button

    def _create_entry(self, parent):
        entry = tk.Entry(parent, bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'])
        entry.pack(side=tk.LEFT, padx=10, pady=5)
        entry.bind('<Return>', lambda event: self.jump_to_segment())
        return entry

    def _bind_events(self):
        self.root.bind('<Key>', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_bpm_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_bpm_hover)
        self.span1 = SpanSelector(self.axs[0, 0], self.on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=self.palette.current_palette['span_color']))
        self.span2 = SpanSelector(self.axs[1, 0], self.on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=self.palette.current_palette['span_color']))
        self.span3 = SpanSelector(self.axs[0, 1], self.on_bpm_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=self.palette.current_palette['span_color']))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def plot_segment(self, index, xlim=None, bpm_xlim=None):
        if index >= len(self.segment_data):
            index = 0
        segment, ref_ecg, r_peaks = self.segment_data[index]
        time_axis = np.linspace(index * self.seconds, (index + 1) * self.seconds, len(segment))
        self._plot_raw_segment(time_axis, segment, r_peaks, xlim)
        self._plot_preprocessed_segment(time_axis, ref_ecg, r_peaks, xlim)
        self._plot_bpm(index, bpm_xlim)
        self._plot_bpm_info(index)
        self.canvas.draw()

    def _plot_raw_segment(self, time_axis, segment, r_peaks, xlim):
        self.axs[0, 0].cla()
        self.axs[0, 0].plot(time_axis, segment, color=self.palette.current_palette['line_color'])
        self.axs[0, 0].plot(time_axis[r_peaks], segment[r_peaks], 'x', color=self.palette.current_palette['peak_color'])
        self.axs[0, 0].set_title(f'Segment {self.current_segment_index} (Raw)')
        self.axs[0, 0].set_xlabel('Time (s)')
        self.axs[0, 0].set_ylabel('Amplitude')
        signal_start, signal_end = time_axis[0], time_axis[-1]
        if xlim:
            xlim = (max(xlim[0], signal_start), min(xlim[1], signal_end))
            self.axs[0, 0].set_xlim(xlim)
        else:
            self.axs[0, 0].set_xlim(signal_start, signal_end)
        self._apply_plot_style(self.axs[0, 0])

    def _plot_preprocessed_segment(self, time_axis, ref_ecg, r_peaks, xlim):
        self.axs[1, 0].cla()
        self.axs[1, 0].plot(time_axis, ref_ecg, color=self.palette.current_palette['line_color'])
        self.axs[1, 0].plot(time_axis[r_peaks], ref_ecg[r_peaks], 'x', color=self.palette.current_palette['peak_color'])
        self.axs[1, 0].set_title(f'Segment {self.current_segment_index} (Preprocessed)')
        self.axs[1, 0].set_xlabel('Time (s)')
        self.axs[1, 0].set_ylabel('Amplitude')
        signal_start, signal_end = time_axis[0], time_axis[-1]
        if xlim:
            xlim = (max(xlim[0], signal_start), min(xlim[1], signal_end))
            self.axs[1, 0].set_xlim(xlim)
        else:
            self.axs[1, 0].set_xlim(signal_start, signal_end)
        self._apply_plot_style(self.axs[1, 0])

    def _plot_bpm(self, index, bpm_xlim):
        self.axs[0, 1].cla()
        self.axs[0, 1].plot(self.bpm_list, color=self.palette.current_palette['line_color'], label='Segment BPM')
        self.axs[0, 1].axhline(y=np.mean(self.bpm_list), color=self.palette.current_palette['bpm_color'], label='Mean BPM', alpha=0.9, linewidth=3)
        self.axs[0, 1].plot(self.bpm_reg, color=self.palette.current_palette['peak_color'], label='Regression Line')
        self.axs[0, 1].legend()
        self._highlight_bpm_segment(index)
        self.axs[0, 1].set_title('BPM over time')
        self.axs[0, 1].set_xlabel(f'Segments (length {self.seconds}s)')
        self.axs[0, 1].set_ylabel('BPM')
        if bpm_xlim:
            self.axs[0, 1].set_xlim(bpm_xlim)
        self._apply_plot_style(self.axs[0, 1])

    def _highlight_bpm_segment(self, index):
        if self.bpm_tag:
            first_tag = self.bpm_tag[0]
            last_tag = self.bpm_tag[-1]
            if first_tag == last_tag:
                self.axs[0, 1].axvline(x=first_tag, color=self.palette.current_palette['peak_color'], alpha=0.2)
            else:
                self.axs[0, 1].axvspan(first_tag - 1, last_tag, color=self.palette.current_palette['peak_color'], alpha=0.2)
        self.axs[0, 1].axvspan(index - 0.5, index + 0.5, color=self.palette.current_palette['highlight_color'], alpha=self.palette.current_palette['highlight_alpha'])

    def _plot_bpm_info(self, index):
        self.axs[1, 1].cla()
        valid_bpm_list = self._get_valid_bpm_list()
        mean_bpm = np.mean(valid_bpm_list)
        self.axs[1, 1].text(0.5, 0.5, f'Predicted segment BPM:\n{self.bpm_list[index]:.2f}\n\n\n\nPredicted valid signal BPM:\n{mean_bpm:.2f}', fontsize=14, ha='center', va='center', color=self.palette.current_palette['fg'], transform=self.axs[1, 1].transAxes)
        self.axs[1, 1].set_axis_off()

    def _get_valid_bpm_list(self):
        if self.bpm_tag:
            valid_bpm_list = []
            start = 0
            for i, tag in enumerate(self.bpm_tag):
                if i % 2 == 0:
                    valid_bpm_list.extend(self.bpm_list[start:tag])
                start = tag + 1
            valid_bpm_list.extend(self.bpm_list[start:])
        else:
            valid_bpm_list = self.bpm_list
        return valid_bpm_list

    def _apply_plot_style(self, ax):
        palette = self.palette.current_palette
        ax.set_facecolor(palette['plot_bg'])
        ax.tick_params(colors=palette['plot_fg'])
        ax.yaxis.label.set_color(palette['plot_fg'])
        ax.xaxis.label.set_color(palette['plot_fg'])
        ax.title.set_color(palette['title_color'])
        self.fig.patch.set_facecolor(palette['plot_bg'])
        self.fig.patch.set_alpha(1.0)

    def on_key(self, event):
        bpm_xlim = self.axs[0, 1].get_xlim()
        if event.keysym in ['d', 'Right'] and self.current_segment_index < len(self.segment_data) - 1:
            self.current_segment_index += 1
            self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim)
        elif event.keysym in ['a', 'Left'] and self.current_segment_index > 0:
            self.current_segment_index -= 1
            self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim)

    def on_select(self, xmin, xmax):
        if xmin == xmax:
            xlim = self.axs[0, 0].get_xlim()
        else:
            xlim = (xmin, xmax)
        bpm_xlim = self.axs[0, 1].get_xlim()
        self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim, xlim=xlim)

    def on_bpm_select(self, xmin, xmax):
        bpm_xlim = (xmin, xmax) if xmax - xmin else self.axs[0, 1].get_xlim()
        xlim = self.axs[0, 0].get_xlim()
        self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim, xlim=xlim)

    def reset_zoom(self):
        self.plot_segment(self.current_segment_index)

    def toggle_night_mode(self):
        self.palette.toggle_night_mode()
        self.root.configure(bg=self.palette.current_palette['bg'])
        self._update_button_styles()
        self.plot_segment(self.current_segment_index)

    def _update_button_styles(self):
        self.reset_button.configure(bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'])
        self.night_mode_button.configure(bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'], text="Toggle Day Mode" if self.palette.is_night_mode else "Toggle Night Mode")
        self.button_frame.configure(bg=self.palette.current_palette['bg'])
        self.segment_entry.configure(bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'])
        self.jump_button.configure(bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'])
        self.hint_button.configure(bg=self.palette.current_palette['button_bg'], fg=self.palette.current_palette['button_fg'])

    def on_bpm_click(self, event):
        if event.dblclick and event.inaxes == self.axs[0, 1]:
            x = int(event.xdata)
            if 0 <= x < len(self.segment_data):
                self.current_segment_index = x
                bpm_xlim = self.axs[0, 1].get_xlim()
                self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim)

    def on_bpm_hover(self, event):
        if event.inaxes == self.axs[0, 1]:
            x = int(event.xdata)
            if 0 <= x < len(self.segment_data):
                if not hasattr(self, 'last_x') or self.last_x != x:
                    self.last_x = x
                    self._update_hover_line(x)

    def _update_hover_line(self, x):
        for line in self.axs[0, 1].get_lines():
            if line.get_linestyle() == '--':
                line.remove()
        for text in self.axs[0, 1].texts:
            if not text.get_text() == 'Invalid Signal':
                text.remove()
        self.axs[0, 1].axvline(x=x, color=self.palette.current_palette['hover_line_color'], linestyle='--')
        self.axs[0, 1].text(x, self.axs[0, 1].get_ylim()[1], f'{x}', color=self.palette.current_palette['hover_text_color'], verticalalignment='top')
        self.canvas.draw_idle()

    def jump_to_segment(self):
        try:
            index = int(self.segment_entry.get())
            if 0 <= index < len(self.segment_data):
                self.current_segment_index = index
                bpm_xlim = self.axs[0, 1].get_xlim()
                self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim)
                self.segment_entry.delete(0, tk.END)
        except ValueError:
            pass

    def show_hint(self):
        hint_message = (
            "Instructions:\n"
            "- Use 'a' or 'Left Arrow' to go to the previous segment.\n"
            "- Use 'd' or 'Right Arrow' to go to the next segment.\n"
            "- Double-click on the BPM plot to jump to a specific segment.\n"
            "- Click and drag over any plot to zoom in on a specific range.\n"
            "- Use the 'Reset Zoom' button to reset the zoom level.\n"
            "- Use the 'Toggle Night Mode' button to switch between day and night modes.\n"
            "- Enter a segment number and press 'Jump to Segment' to jump to a specific segment."
        )
        hint_window = tk.Toplevel(self.root)
        hint_window.title("Hint")
        hint_label = tk.Label(hint_window, text=hint_message, justify=tk.LEFT)
        hint_label.pack(padx=10, pady=10)
        hint_window.transient(self.root)
        hint_window.grab_set()
        hint_window.focus_set()

    def on_closing(self):
        self.config_manager.set("last_index", self.current_segment_index)
        self.root.destroy()

class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return json.load(file)
        else:
            return {}

    def save_config(self):
        with open(self.config_file, "w") as file:
            json.dump(self.config, file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

class ECGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Processor")
        self.root.resizable(False, False)

        self.config_manager = ConfigManager("ecg_config.json")

        self._setup_ui()
        self._bind_events()

    def _setup_ui(self):
        self.file_label = tk.Label(self.root, text="Select ECG Data File:")
        self.file_label.pack(pady=5)

        self.file_entry = tk.Entry(self.root, width=50)
        self.file_entry.pack(pady=5)
        self.file_entry.insert(0, self.config_manager.get("last_path", ""))

        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        self.hz_label = tk.Label(self.root, text="Enter Frequency (Hz):")
        self.hz_label.pack(pady=5)

        self.hz_entry = tk.Entry(self.root, width=10)
        self.hz_entry.pack(pady=5)
        self.hz_entry.insert(0, self.config_manager.get("last_hz", ""))

        self.process_button = tk.Button(self.root, text="Process", command=self.validate_and_process)
        self.process_button.pack(pady=10)

        self.mode_var = tk.StringVar(value="peak_detection")
        self.peak_detection_radio = tk.Radiobutton(self.root, text="Peak Detection", variable=self.mode_var, value="peak_detection", command=self.toggle_dropdown)
        self.peak_detection_radio.pack(pady=5)
        self.quality_classifier_radio = tk.Radiobutton(self.root, text="Quality Classifier", variable=self.mode_var, value="quality_classifier", command=self.toggle_dropdown)
        self.quality_classifier_radio.pack(pady=5)

        self.cluster_frame = tk.Frame(self.root)
        self.cluster_label = tk.Label(self.cluster_frame, text="Select Number of Clusters:")
        self.cluster_label.pack(side=tk.LEFT, padx=5)
        self.cluster_var = tk.IntVar(value=3)
        self.cluster_dropdown = tk.OptionMenu(self.cluster_frame, self.cluster_var, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.cluster_dropdown.pack(side=tk.LEFT, padx=5)

        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress, maximum=100)
        self.progress_bar.pack(pady=10, fill=tk.X)

        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack(pady=0)

    def _bind_events(self):
        self.hz_entry.bind('<Return>', self.validate_and_process)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_dropdown(self):
        if self.mode_var.get() == "quality_classifier":
            self.cluster_frame.pack(pady=5)
        else:
            self.cluster_frame.pack_forget()

    def validate_and_process(self, _=None):
        try:
            hz = self.hz_entry.get()
            if not hz:
                raise ValueError("Frequency must be specified")
            elif int(hz) <= 0:
                raise ValueError("Frequency must be positive")
            self.process_ecg()
        except ValueError as e:
            self.progress_label.config(text=f"Invalid frequency: {e}")

    def browse_file(self):
        self.progress_label.config(text="")
        self.progress.set(0)
        initial_dir = os.path.dirname(self.config_manager.get("last_path", ""))
        file_path = tk.filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("ECG Data Files", "*.dat")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def process_ecg(self):
        self._disable_ui()

        self.progress.set(0)
        self.progress_label.config(text="Loading data...")
        data_path = self.file_entry.get()
        hz = int(self.hz_entry.get())
        seconds = 10

        self.config_manager.set("last_path", data_path)
        self.config_manager.set("last_hz", hz)

        if self.mode_var.get() == "peak_detection":
            threading.Thread(target=self._run_detector, args=(data_path, hz, seconds)).start()
        elif self.mode_var.get() == "quality_classifier":
            threading.Thread(target=self._run_clusterer, args=(data_path, hz, seconds)).start()

    def _run_detector(self, data_path, hz, seconds):
        self.ecg_processor = ECGPeakDetector(data_path, hz, seconds)
        ecg_signal = self.ecg_processor.load_data()
        segments = np.array_split(ecg_signal, len(ecg_signal) // (self.ecg_processor.hz * self.ecg_processor.seconds))

        total_steps = 100
        step_interval = max(1, len(segments) // total_steps)

        for i, segment in enumerate(segments):
            self.progress_label.config(text=f"Detecting peaks {i + 1}/{len(segments)}")
            ref_ecg = self.ecg_processor.pan_tompkins(i, segment)
            r_peaks = self.ecg_processor.detect_r_peaks(ref_ecg)
            bpm = len(r_peaks) / self.ecg_processor.seconds * 60
            self.ecg_processor.bpm_list.append(bpm)
            self.ecg_processor.segment_data.append((segment, ref_ecg, r_peaks))
            if i % step_interval == 0:
                self.progress.set((i / len(segments)) * 100)
                self.root.update_idletasks()

        self._post_process_detector()

    def _post_process_detector(self):
        filtered_bpm_tag = []
        segment_start = None

        for i in range(len(self.ecg_processor.bpm_tag)):
            if segment_start is None:
                segment_start = self.ecg_processor.bpm_tag[i]
            if i == len(self.ecg_processor.bpm_tag) - 1 or self.ecg_processor.bpm_tag[i + 1] - self.ecg_processor.bpm_tag[i] > 50:
                if i - self.ecg_processor.bpm_tag.index(segment_start) >= 2:
                    filtered_bpm_tag.append(segment_start)
                    filtered_bpm_tag.append(self.ecg_processor.bpm_tag[i])
                segment_start = None

        self.ecg_processor.bpm_tag = filtered_bpm_tag

        if self.ecg_processor.bpm_tag:
            bpm_list = []
            start = 0
            for tag in self.ecg_processor.bpm_tag:
                bpm_list.extend(self.ecg_processor.bpm_list[start:tag])
                start = tag
            bpm_list.extend(self.ecg_processor.bpm_list[start:])
        else:
            bpm_list = self.ecg_processor.bpm_list

        self.ecg_processor.bpm_reg = []
        if self.ecg_processor.bpm_tag:
            start = 0
            for tag in self.ecg_processor.bpm_tag:
                bpm_segment = bpm_list[start:tag]
                if bpm_segment:
                    bpm_reg_segment = self.ecg_processor.polymer_regression(np.arange(len(bpm_segment)), bpm_segment)
                    self.ecg_processor.bpm_reg.extend(bpm_reg_segment)
                start = tag
            bpm_segment = bpm_list[start:]
            if bpm_segment:
                if len(bpm_segment) <= 2:
                    bpm_reg_segment = [np.mean(bpm_segment)] * len(bpm_segment)
                else:
                    bpm_reg_segment = self.ecg_processor.polymer_regression(np.arange(len(bpm_segment)), bpm_segment)
                self.ecg_processor.bpm_reg.extend(bpm_reg_segment)
        else:
            self.ecg_processor.bpm_reg = self.ecg_processor.polymer_regression(np.arange(len(bpm_list)), bpm_list)

        self.progress.set(100)
        self.progress_label.config(text="Processing complete")
        self.root.after(0, self._show_peak_viewer)

        self._enable_ui()

    def _run_clusterer(self, data_path, hz, seconds):
        self.ecg_processor = ECGClusterer(data_path, hz, seconds)
        ecg_signal = self.ecg_processor.load_data()
        self.progress_label.config(text="Preprocessing data...")
        self.progress.set(10)
        self.root.update_idletasks()

        features = self.ecg_processor.preprocess_data(ecg_signal)
        n_clusters = self.cluster_var.get()

        self.progress_label.config(text="Clustering data...")
        self.progress.set(50)
        self.root.update_idletasks()

        labels = self.ecg_processor.kmeans_clustering(features, n_clusters)

        self.progress.set(100)
        self.progress_label.config(text="Processing complete")
        self.root.after(0, self._show_cluster_viewer, ecg_signal, features, labels, hz, seconds)

        self._enable_ui()

    def _show_peak_viewer(self):
        segment_data = self.ecg_processor.segment_data
        bpm_list = self.ecg_processor.bpm_list
        bpm_reg = self.ecg_processor.bpm_reg
        bpm_tag = self.ecg_processor.bpm_tag

        viewer_root = tk.Toplevel(self.root)
        viewer_root.title("ECG Segment Viewer")
        ecg_viewer = ECGSegmentViewer(viewer_root, segment_data, bpm_list, bpm_reg, bpm_tag, 10, self.config_manager, self.file_entry.get())
        ecg_viewer.current_segment_index = self.config_manager.get("last_index", 0)
        ecg_viewer.plot_segment(ecg_viewer.current_segment_index)

    def _show_cluster_viewer(self, ecg_signal, features, labels, hz, seconds):
        viewer = ECGClusterViewer(tk.Toplevel(self.root), ecg_signal, features, labels, hz, seconds)

    def _enable_ui(self):
        self.file_entry.config(state=tk.NORMAL)
        self.browse_button.config(state=tk.NORMAL)
        self.hz_entry.config(state=tk.NORMAL)
        self.process_button.config(state=tk.NORMAL)
        self.peak_detection_radio.config(state=tk.NORMAL)
        self.quality_classifier_radio.config(state=tk.NORMAL)
        self.cluster_dropdown.config(state=tk.NORMAL)
    
    def _disable_ui(self):
        self.file_entry.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.hz_entry.config(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        self.peak_detection_radio.config(state=tk.DISABLED)
        self.quality_classifier_radio.config(state=tk.DISABLED)
        self.cluster_dropdown.config(state=tk.DISABLED)

    def on_closing(self):
        if hasattr(self, 'ecg_viewer'):
            self.config_manager.set("last_index", self.ecg_viewer.current_segment_index)
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
