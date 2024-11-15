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

    # Dynamic thresholding
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

    res = []
    
    for i, peak in enumerate(r_peaks):
        distance = peak - r_peaks[i-1] if i > 0 else np.mean(RR_intervals)
        if distance < 0.7 * np.mean(RR_intervals):
            peak_amplitude = filtered_processed_signal[peak]
            mean_amplitude = np.mean(filtered_processed_signal[r_peaks])
            if peak_amplitude < 0.5 * mean_amplitude:
                print(f"Suspicious peak amplitude detected at index {i} with amplitude {peak_amplitude}")
                continue
        res.append(peak)

    return np.array(res)

def polymer_regression(x, y):
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

# Load and process the ECG signal
ecg_signal = load_data(data_path)
segments = np.array_split(ecg_signal, len(ecg_signal) // (hz * seconds))
bpm_list = []

old = 0
old_segment = None
old_r_peaks = None
old_ref_ecg = None

segment_data = []

for segment in segments[0:500]: #change later to segments
    ref_ecg = refined_pan_tompkins_ecg_processing(segment, hz)
    r_peaks = detect_r_peaks_with_refined_threshold(ref_ecg, hz)
    bpm = len(r_peaks) / seconds * 60
    bpm_list.append(bpm)
    segment_data.append((segment, ref_ecg, r_peaks))
    old = bpm

bpm_reg = polymer_regression(np.arange(len(bpm_list)), bpm_list)
# Create a Tkinter window
root = tk.Tk()
root.title("ECG Segment Viewer")

fig, axs = plt.subplots(2, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

current_segment_index = 0
is_night_mode = False

# Define color palettes
day_palette = {
    'bg': 'white',
    'fg': 'black',
    'plot_bg': 'white',
    'plot_fg': 'black',
    'line_color': 'blue',
    'peak_color': 'red',  # Color for peak markers
    'button_bg': 'white',
    'button_fg': 'black',
    'title_color': 'black',
    'span_color': 'red',
    'hover_line_color': 'gray',
    'hover_text_color': 'gray',
    'highlight_color': 'red',
    'highlight_alpha': 0.3
}

night_palette = {
    'bg': '#2E2E2E',  # Dark gray background
    'fg': '#E0E0E0',  # Light gray foreground
    'plot_bg': '#1C1C1C',  # Almost black plot background
    'plot_fg': '#F5F5F5',  # Off-white plot foreground
    'line_color': '#00FF00',  # Bright green line color
    'peak_color': '#FF4500',  # Bright orange color for peak markers
    'button_bg': '#3A3A3A',  # Slightly lighter gray for buttons
    'button_fg': '#E0E0E0',  # Light gray button text
    'title_color': '#F5F5F5',  # Off-white title color
    'span_color': 'red',
    'hover_line_color': 'gray',
    'hover_text_color': 'gray',
    'highlight_color': 'red',
    'highlight_alpha': 0.3
}

current_palette = day_palette

def plot_segment(index, xlim=None, bpm_xlim=None):
    segment, ref_ecg, r_peaks = segment_data[index]
    time_axis = np.linspace(index * seconds, (index + 1) * seconds, len(segment))
    axs[0, 0].cla()
    axs[1, 0].cla()
    palette = current_palette
    axs[0, 0].plot(time_axis, segment, color=palette['line_color'])
    axs[0, 0].plot(time_axis[r_peaks], segment[r_peaks], 'x', color=palette['peak_color'])
    axs[0, 0].set_title(f'Segment {index} (Raw)')
    axs[1, 0].plot(time_axis, ref_ecg, color=palette['line_color'])
    axs[1, 0].plot(time_axis[r_peaks], ref_ecg[r_peaks], 'x', color=palette['peak_color'])
    axs[1, 0].set_title(f'Segment {index} (Preprocessed)')
    axs[1, 0].set_xlabel('Time (s)')
    if xlim:
        axs[0, 0].set_xlim(xlim)
        axs[1, 0].set_xlim(xlim)
    
    axs[0, 0].set_facecolor(palette['plot_bg'])
    axs[0, 0].tick_params(colors=palette['plot_fg'])
    axs[0, 0].yaxis.label.set_color(palette['plot_fg'])
    axs[0, 0].xaxis.label.set_color(palette['plot_fg'])
    axs[0, 0].title.set_color(palette['title_color'])
    axs[1, 0].set_facecolor(palette['plot_bg'])
    axs[1, 0].tick_params(colors=palette['plot_fg'])
    axs[1, 0].yaxis.label.set_color(palette['plot_fg'])
    axs[1, 0].xaxis.label.set_color(palette['plot_fg'])
    axs[1, 0].title.set_color(palette['title_color'])
    fig.patch.set_facecolor(palette['plot_bg'])
    fig.patch.set_alpha(1.0)
    
    # Plot BPM list
    axs[0, 1].cla()
    axs[0, 1].plot(bpm_list, color=palette['line_color'])
    axs[0, 1].plot(bpm_reg, color=palette['peak_color'])
    axs[0, 1].axvspan(index - 0.5, index + 0.5, color=palette['highlight_color'], alpha=palette['highlight_alpha'])
    axs[0, 1].set_title('BPM')
    print(bpm_xlim, index)
    if bpm_xlim:
        axs[0, 1].set_xlim(bpm_xlim)
    axs[0, 1].set_facecolor(palette['plot_bg'])
    axs[0, 1].tick_params(colors=palette['plot_fg'])
    axs[0, 1].yaxis.label.set_color(palette['plot_fg'])
    axs[0, 1].xaxis.label.set_color(palette['plot_fg'])
    axs[0, 1].title.set_color(palette['title_color'])
    # Display current BPM
    axs[1, 1].cla()
    axs[1, 1].text(0.5, 0.5, f'Predicted BPM:\n{bpm_list[index]:.2f}', fontsize=14, ha='center', color=current_palette['fg'])
    axs[1, 1].set_axis_off()
    canvas.draw()

def on_key(event):
    global current_segment_index
    bpm_xlim = axs[0, 1].get_xlim()
    if event.keysym in ['d', 'Right'] and current_segment_index < len(segment_data) - 1:
        current_segment_index += 1
        plot_segment(current_segment_index, bpm_xlim=bpm_xlim)
    elif event.keysym in ['a', 'Left'] and current_segment_index > 0:
        current_segment_index -= 1
        plot_segment(current_segment_index, bpm_xlim=bpm_xlim)

def on_select(xmin, xmax):
    bpm_xlim = axs[0, 1].get_xlim()
    plot_segment(current_segment_index, bpm_xlim=bpm_xlim ,xlim=(xmin, xmax))

def on_bpm_select(xmin, xmax):
    xlim = axs[0, 0].get_xlim()
    plot_segment(current_segment_index, bpm_xlim=(xmin, xmax), xlim=xlim)

def reset_zoom():
    plot_segment(current_segment_index)

def toggle_night_mode():
    global is_night_mode, current_palette
    is_night_mode = not is_night_mode
    current_palette = night_palette if is_night_mode else day_palette
    root.configure(bg=current_palette['bg'])
    reset_button.configure(bg=current_palette['button_bg'], fg=current_palette['button_fg'])
    night_mode_button.configure(bg=current_palette['button_bg'], fg=current_palette['button_fg'], text="Toggle Day Mode" if is_night_mode else "Toggle Night Mode")
    plot_segment(current_segment_index)

def on_bpm_click(event):
    global current_segment_index
    if event.dblclick and event.inaxes == axs[0, 1]:
        x = int(event.xdata)
        if 0 <= x < len(segment_data):
            current_segment_index = x
            bpm_xlim = axs[0, 1].get_xlim()
            plot_segment(current_segment_index, bpm_xlim=bpm_xlim)

def on_bpm_hover(event):
    if event.inaxes == axs[0, 1]:
        for line in axs[0, 1].get_lines():
            if line.get_linestyle() == '--':
                line.remove()  # Clear previous hover lines
        for text in axs[0, 1].texts:
            text.remove()  # Clear previous hover text
        x = int(event.xdata)
        if 0 <= x < len(segment_data):
            axs[0, 1].axvline(x=x, color=current_palette['hover_line_color'], linestyle='--')
            axs[0, 1].text(x, axs[0, 1].get_ylim()[1], f'{x}', color=current_palette['hover_text_color'], verticalalignment='top')
        canvas.draw_idle()

root.bind('<Key>', on_key)
fig.canvas.mpl_connect('button_press_event', on_bpm_click)
fig.canvas.mpl_connect('motion_notify_event', on_bpm_hover)

plot_segment(current_segment_index)

span1 = SpanSelector(axs[0, 0], on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=current_palette['span_color']))
span2 = SpanSelector(axs[1, 0], on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=current_palette['span_color']))
span3 = SpanSelector(axs[0, 1], on_bpm_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=current_palette['span_color']))

reset_button = tk.Button(root, text="Reset Zoom", command=reset_zoom)
reset_button.pack(side=tk.BOTTOM)

night_mode_button = tk.Button(root, text="Toggle Night Mode", command=toggle_night_mode)
night_mode_button.pack(side=tk.BOTTOM)

def on_closing():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
