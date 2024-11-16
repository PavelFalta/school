import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, detrend
import scipy.signal as signal
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from tkinter import ttk
import os
import json

class ECGProcessor:
    def __init__(self, data_path, hz, seconds):
        self.data_path = data_path
        self.hz = hz
        self.seconds = seconds
        self.segment_data = []
        self.bpm_list = []
        self.bpm_reg = None

    def load_data(self):
        data = np.fromfile(self.data_path, dtype=np.int16)
        return data

    def refined_pan_tompkins_ecg_processing(self, ecg_signal, lowcut=1, highcut=30, filter_order=2, window_duration=0.12):
        def bandpass_filter(signal, lowcut, highcut, fs, order):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal)

        filtered_signal = bandpass_filter(ecg_signal, lowcut, highcut, self.hz, filter_order)
        differentiated_signal = np.diff(filtered_signal, prepend=filtered_signal[0])
        squared_signal = differentiated_signal ** 2
        window_size = int(window_duration * self.hz)
        mwi_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

        return mwi_signal

    def detect_r_peaks_with_refined_threshold(self, processed_signal, initial_threshold_factor=0.5):
        min_distance = int(self.hz * 60 / 250)  # Minimum distance corresponding to 250 BPM

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

        filtered_processed_signal = lowpass_filter(processed_signal, 15, self.hz)

        # Initial peak detection
        peaks, properties = find_peaks(filtered_processed_signal, height=initial_threshold, distance=min_distance)

        # Dynamic thresholding
        signal_peak = 0
        noise_peak = 0
        threshold_high = initial_threshold
        threshold_low = 0.5 * threshold_high
        rr_intervals = []
        r_peaks = []

        for i, peak in enumerate(peaks):
            if filtered_processed_signal[peak] > threshold_high:
                signal_peak = 0.125 * filtered_processed_signal[peak] + 0.875 * signal_peak
                r_peaks.append(peak)
                if len(r_peaks) > 1:
                    rr_intervals.append(r_peaks[-1] - r_peaks[-2])
            else:
                noise_peak = 0.125 * filtered_processed_signal[peak] + 0.875 * noise_peak

            threshold_high = noise_peak + 0.25 * (signal_peak - noise_peak)

        res = []
        
        for i, peak in enumerate(r_peaks):
            distance = peak - r_peaks[i-1] if i > 0 else np.mean(rr_intervals)
            if distance < 0.7 * np.mean(rr_intervals):
                peak_amplitude = filtered_processed_signal[peak]
                mean_amplitude = np.mean(filtered_processed_signal[r_peaks])
                if peak_amplitude < 0.5 * mean_amplitude:
                    continue
            res.append(peak)

        return np.array(res)

    def polymer_regression(self, x, y):
        best_r2 = -np.inf
        best_y_pred = None
        best_degree = 1
        no_improvement_count = 0
        degree = 1

        while no_improvement_count < 10:
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

    def process(self):
        ecg_signal = self.load_data()
        segments = np.array_split(ecg_signal, len(ecg_signal) // (self.hz * self.seconds))

        for segment in segments:
            ref_ecg = self.refined_pan_tompkins_ecg_processing(segment)
            r_peaks = self.detect_r_peaks_with_refined_threshold(ref_ecg)
            bpm = len(r_peaks) / self.seconds * 60
            self.bpm_list.append(bpm)
            self.segment_data.append((segment, ref_ecg, r_peaks))

        self.bpm_reg = self.polymer_regression(np.arange(len(self.bpm_list)), self.bpm_list)


class ECGSegmentViewer:
    def __init__(self, root, segment_data, bpm_list, bpm_reg, seconds, config_manager):
        self.root = root
        self.segment_data = segment_data
        self.bpm_list = bpm_list
        self.bpm_reg = bpm_reg
        self.seconds = seconds
        self.config_manager = config_manager
        self.current_segment_index = 0
        self.is_night_mode = False
        self.current_palette = self.day_palette

        self.fig, self.axs = plt.subplots(2, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1.5]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.root.bind('<Key>', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_bpm_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_bpm_hover)

        self.default = None
        self.plot_segment(self.current_segment_index)
        self.default = self.axs[0, 1].get_xlim()

        self.span1 = SpanSelector(self.axs[0, 0], self.on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=self.current_palette['span_color']))
        self.span2 = SpanSelector(self.axs[1, 0], self.on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=self.current_palette['span_color']))
        self.span3 = SpanSelector(self.axs[0, 1], self.on_bpm_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=self.current_palette['span_color']))

        self.button_frame = tk.Frame(self.root, bg=self.current_palette['bg'])
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.reset_button = tk.Button(self.button_frame, text="Reset Zoom", command=self.reset_zoom, bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.reset_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.night_mode_button = tk.Button(self.button_frame, text="Toggle Night Mode", command=self.toggle_night_mode, bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.night_mode_button.pack(side=tk.RIGHT, padx=10, pady=5)

        self.segment_entry = tk.Entry(self.button_frame, bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.segment_entry.pack(side=tk.LEFT, padx=10, pady=5)
        self.segment_entry.bind('<Return>', lambda event: self.jump_to_segment())

        self.jump_button = tk.Button(self.button_frame, text="Jump to Segment", command=self.jump_to_segment, bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.jump_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.hint_button = tk.Button(self.button_frame, text="?", command=self.show_hint, bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.hint_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.hint_button.bind("<Button-1>", lambda event: self.hint_button.lift())

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    day_palette = {
        'bg': 'white',
        'fg': 'black',
        'plot_bg': 'white',
        'plot_fg': 'black',
        'line_color': 'blue',
        'peak_color': 'red',
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
        'button_bg': '#2E2E2E',
        'button_fg': '#C0C0C0',
        'title_color': '#D3D3D3',
        'span_color': '#FF4500',
        'hover_line_color': '#808080',
        'hover_text_color': '#808080',
        'highlight_color': '#FF4500',
        'highlight_alpha': 0.4
    }

    def plot_segment(self, index, xlim=None, bpm_xlim=None):
        segment, ref_ecg, r_peaks = self.segment_data[index]
        time_axis = np.linspace(index * self.seconds, (index + 1) * self.seconds, len(segment))
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        palette = self.current_palette
        self.axs[0, 0].plot(time_axis, segment, color=palette['line_color'])
        self.axs[0, 0].plot(time_axis[r_peaks], segment[r_peaks], 'x', color=palette['peak_color'])
        self.axs[0, 0].set_title(f'Segment {index} (Raw)')
        self.axs[0, 0].set_xlabel('Time (s)')
        self.axs[1, 0].plot(time_axis, ref_ecg, color=palette['line_color'])
        self.axs[1, 0].plot(time_axis[r_peaks], ref_ecg[r_peaks], 'x', color=palette['peak_color'])
        self.axs[1, 0].set_title(f'Segment {index} (Preprocessed)')
        self.axs[1, 0].set_xlabel('Time (s)')

        signal_start = time_axis[0]
        signal_end = time_axis[-1]
        if xlim:
            xlim = (max(xlim[0], signal_start), min(xlim[1], signal_end))
            self.axs[0, 0].set_xlim(xlim)
            self.axs[1, 0].set_xlim(xlim)
        else:
            self.axs[0, 0].set_xlim(signal_start, signal_end)
            self.axs[1, 0].set_xlim(signal_start, signal_end)

        self.axs[0, 0].set_facecolor(palette['plot_bg'])
        self.axs[0, 0].tick_params(colors=palette['plot_fg'])
        self.axs[0, 0].yaxis.label.set_color(palette['plot_fg'])
        self.axs[0, 0].xaxis.label.set_color(palette['plot_fg'])
        self.axs[0, 0].title.set_color(palette['title_color'])
        self.axs[1, 0].set_facecolor(palette['plot_bg'])
        self.axs[1, 0].tick_params(colors=palette['plot_fg'])
        self.axs[1, 0].yaxis.label.set_color(palette['plot_fg'])
        self.axs[1, 0].xaxis.label.set_color(palette['plot_fg'])
        self.axs[1, 0].title.set_color(palette['title_color'])
        self.fig.patch.set_facecolor(palette['plot_bg'])
        self.fig.patch.set_alpha(1.0)

        self.axs[0, 1].cla()
        self.axs[0, 1].plot(self.bpm_list, color=palette['line_color'], label='Segment BPM')
        self.axs[0, 1].plot(self.bpm_reg, color=palette['peak_color'], label='Regression Line')
        self.axs[0, 1].legend()
        self.axs[0, 1].axvspan(index - 0.5, index + 0.5, color=palette['highlight_color'], alpha=palette['highlight_alpha'])
        self.axs[0, 1].set_title('BPM over time')
        self.axs[0, 1].set_xlabel(f'Segments (length {self.seconds}s)')

        if bpm_xlim:
            self.axs[0, 1].set_xlim(bpm_xlim)
            if self.default and not bpm_xlim == self.default:
                if bpm_xlim[0] < index < bpm_xlim[1]:
                    length = bpm_xlim[1] - bpm_xlim[0]
                    middle = index
                    self.axs[0, 1].set_xlim(middle - length / 2, middle + length / 2)
        self.axs[0, 1].set_facecolor(palette['plot_bg'])
        self.axs[0, 1].tick_params(colors=palette['plot_fg'])
        self.axs[0, 1].yaxis.label.set_color(palette['plot_fg'])
        self.axs[0, 1].xaxis.label.set_color(palette['plot_fg'])
        self.axs[0, 1].title.set_color(palette['title_color'])

        self.axs[1, 1].cla()
        self.axs[1, 1].text(0.5, 0.5, f'Predicted segment BPM:\n{self.bpm_list[index]:.2f}\n\n\n\nPredicted signal BPM:\n{np.mean(self.bpm_list):.2f}', fontsize=14, ha='center', va='center', color=self.current_palette['fg'], transform=self.axs[1, 1].transAxes)
        self.axs[1, 1].set_axis_off()
        self.canvas.draw()

        if not self.default:
            self.fig.tight_layout()

    def on_key(self, event):
        bpm_xlim = self.axs[0, 1].get_xlim()
        if event.keysym in ['d', 'Right'] and self.current_segment_index < len(self.segment_data) - 1:
            self.current_segment_index += 1
            self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim)
        elif event.keysym in ['a', 'Left'] and self.current_segment_index > 0:
            self.current_segment_index -= 1
            self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim)

    def on_select(self, xmin, xmax):
        bpm_xlim = self.axs[0, 1].get_xlim()
        self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim, xlim=(xmin, xmax))

    def on_bpm_select(self, xmin, xmax):
        if not xmax - xmin:
            bpm_xlim = self.axs[0, 1].get_xlim()
        else:
            bpm_xlim = (xmin, xmax)
        xlim = self.axs[0, 0].get_xlim()
        self.plot_segment(self.current_segment_index, bpm_xlim=bpm_xlim, xlim=xlim)

    def reset_zoom(self):
        self.plot_segment(self.current_segment_index)

    def toggle_night_mode(self):
        self.is_night_mode = not self.is_night_mode
        self.current_palette = self.night_palette if self.is_night_mode else self.day_palette
        self.root.configure(bg=self.current_palette['bg'])
        self.reset_button.configure(bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.night_mode_button.configure(bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'], text="Toggle Day Mode" if self.is_night_mode else "Toggle Night Mode")
        self.button_frame.configure(bg=self.current_palette['bg'])
        self.segment_entry.configure(bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.jump_button.configure(bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.hint_button.configure(bg=self.current_palette['button_bg'], fg=self.current_palette['button_fg'])
        self.plot_segment(self.current_segment_index)

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
                    for line in self.axs[0, 1].get_lines():
                        if line.get_linestyle() == '--':
                            line.remove()
                    for text in self.axs[0, 1].texts:
                        text.remove()
                    self.axs[0, 1].axvline(x=x, color=self.current_palette['hover_line_color'], linestyle='--')
                    self.axs[0, 1].text(x, self.axs[0, 1].get_ylim()[1], f'{x}', color=self.current_palette['hover_text_color'], verticalalignment='top')
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

        self.config_manager = ConfigManager("ecg_config.json")

        self.file_label = tk.Label(root, text="Select ECG Data File:")
        self.file_label.pack(pady=5)

        self.file_entry = tk.Entry(root, width=50)
        self.file_entry.pack(pady=5)
        self.file_entry.insert(0, self.config_manager.get("last_path", ""))

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        self.hz_label = tk.Label(root, text="Enter Frequency (Hz):")
        self.hz_label.pack(pady=5)

        self.hz_entry = tk.Entry(root, width=10)
        self.hz_entry.pack(pady=5)
        self.hz_entry.insert(0, self.config_manager.get("last_hz", ""))
        self.hz_entry.bind('<Return>', lambda event: self.process_button.invoke())

        self.process_button = tk.Button(root, text="Process", command=self.process_ecg)
        self.process_button.pack(pady=10)

        self.progress = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, variable=self.progress, maximum=100)
        self.progress_bar.pack(pady=10, fill=tk.X)

    def browse_file(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("ECG Data Files", "*.dat")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def process_ecg(self):
        self.progress.set(0)
        data_path = self.file_entry.get()
        hz = int(self.hz_entry.get())
        seconds = 10

        self.config_manager.set("last_path", data_path)
        self.config_manager.set("last_hz", hz)

        self.ecg_processor = ECGProcessor(data_path, hz, seconds)
        self.root.after(100, self.update_progress)
        self.ecg_processor.process()

        segment_data = self.ecg_processor.segment_data
        bpm_list = self.ecg_processor.bpm_list
        bpm_reg = self.ecg_processor.bpm_reg

        viewer_root = tk.Toplevel(self.root)
        viewer_root.title("ECG Segment Viewer")
        ecg_viewer = ECGSegmentViewer(viewer_root, segment_data, bpm_list, bpm_reg, seconds, self.config_manager)
        ecg_viewer.current_segment_index = self.config_manager.get("last_index", 0)
        ecg_viewer.plot_segment(ecg_viewer.current_segment_index)

    def update_progress(self):
        # Simulate progress update
        self.progress.set(self.progress.get() + 10)
        if self.progress.get() < 100:
            self.root.after(100, self.update_progress)
        else:
            self.progress.set(100)

    def on_closing(self):
        if hasattr(self, 'ecg_viewer'):
            self.config_manager.set("last_index", self.ecg_viewer.current_segment_index)
        print("Closing")
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ECGApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
