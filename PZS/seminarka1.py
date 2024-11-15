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

data_path = "PZS/data/16265.dat"
hz = 128
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
                #print(f"Suspicious peak amplitude detected at index {i} with amplitude {peak_amplitude}")
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

for segment in segments: #change later to segments
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

fig, axs = plt.subplots(2, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1.5]})
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
    'highlight_alpha': 0.4
}

night_palette = {
    'bg': '#1E1E1E',  # Darker gray background
    'fg': '#C0C0C0',  # Light gray foreground
    'plot_bg': '#121212',  # Almost black plot background
    'plot_fg': '#D3D3D3',  # Light gray plot foreground
    'line_color': '#9370DB',  # Medium purple line color
    'peak_color': '#FF6347',  # Tomato color for peak markers
    'button_bg': '#2E2E2E',  # Dark gray for buttons
    'button_fg': '#C0C0C0',  # Light gray button text
    'title_color': '#D3D3D3',  # Light gray title color
    'span_color': '#FF4500',  # Orange red for span selector
    'hover_line_color': '#808080',  # Gray for hover line
    'hover_text_color': '#808080',  # Gray for hover text
    'highlight_color': '#FF4500',  # Orange red for highlight
    'highlight_alpha': 0.4  # Transparency for highlight
}

legacy_black = {
    'bg': 'black',
    'fg': 'white',
    'plot_bg': 'black',
    'plot_fg': 'white',
    'line_color': 'blue',
    'peak_color': 'red',
    'button_bg': 'black',
    'button_fg': 'white',
    'title_color': 'white',
    'span_color': 'red',
    'hover_line_color': 'gray',
    'hover_text_color': 'gray',
    'highlight_color': 'red',
    'highlight_alpha': 0.4
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
    axs[0, 0].set_xlabel('Time (s)')
    axs[1, 0].plot(time_axis, ref_ecg, color=palette['line_color'])
    axs[1, 0].plot(time_axis[r_peaks], ref_ecg[r_peaks], 'x', color=palette['peak_color'])
    axs[1, 0].set_title(f'Segment {index} (Preprocessed)')
    axs[1, 0].set_xlabel('Time (s)')
    
    # Clamp xlim to where the signal starts and ends
    signal_start = time_axis[0]
    signal_end = time_axis[-1]
    if xlim:
        xlim = (max(xlim[0], signal_start), min(xlim[1], signal_end))
        axs[0, 0].set_xlim(xlim)
        axs[1, 0].set_xlim(xlim)
    else:
        axs[0, 0].set_xlim(signal_start, signal_end)
        axs[1, 0].set_xlim(signal_start, signal_end)
    
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
    axs[0, 1].set_title('BPM over time')
    axs[0, 1].set_xlabel(f'Segments (length {seconds}s)')

    if bpm_xlim:
        axs[0, 1].set_xlim(bpm_xlim)
        if default and not bpm_xlim == default:
            if bpm_xlim[0] < index < bpm_xlim[1]:
                length = bpm_xlim[1] - bpm_xlim[0]
                middle = index
                axs[0, 1].set_xlim(middle - length / 2, middle + length / 2)
    axs[0, 1].set_facecolor(palette['plot_bg'])
    axs[0, 1].tick_params(colors=palette['plot_fg'])
    axs[0, 1].yaxis.label.set_color(palette['plot_fg'])
    axs[0, 1].xaxis.label.set_color(palette['plot_fg'])
    axs[0, 1].title.set_color(palette['title_color'])
    # Display current BPM
    axs[1, 1].cla()
    axs[1, 1].text(0.5, 0.5, f'Predicted segment BPM:\n{bpm_list[index]:.2f}\n\n\n\nPredicted signal BPM:\n{np.mean(bpm_list):.2f}', fontsize=14, ha='center', va='center', color=current_palette['fg'], transform=axs[1, 1].transAxes)
    axs[1, 1].set_axis_off()
    canvas.draw()

    if not default:
        fig.tight_layout()

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
    if not xmax-xmin:
        bpm_xlim = axs[0,1].get_xlim()
    else:
        bpm_xlim = (xmin, xmax)
    xlim = axs[0, 0].get_xlim()
    plot_segment(current_segment_index, bpm_xlim=bpm_xlim, xlim=xlim)

def reset_zoom():
    plot_segment(current_segment_index)

def toggle_night_mode():
    global is_night_mode, current_palette
    is_night_mode = not is_night_mode
    current_palette = night_palette if is_night_mode else day_palette
    root.configure(bg=current_palette['bg'])
    reset_button.configure(bg=current_palette['button_bg'], fg=current_palette['button_fg'])
    night_mode_button.configure(bg=current_palette['button_bg'], fg=current_palette['button_fg'], text="Toggle Day Mode" if is_night_mode else "Toggle Night Mode")
    button_frame.configure(bg=current_palette['bg'])
    segment_entry.configure(bg=current_palette['button_bg'], fg=current_palette['button_fg'])
    jump_button.configure(bg=current_palette['button_bg'], fg=current_palette['button_fg'])
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
        x = int(event.xdata)
        if 0 <= x < len(segment_data):
            if not hasattr(on_bpm_hover, 'last_x') or on_bpm_hover.last_x != x:
                on_bpm_hover.last_x = x
                for line in axs[0, 1].get_lines():
                    if line.get_linestyle() == '--':
                        line.remove()  # Clear previous hover lines
                for text in axs[0, 1].texts:
                    text.remove()  # Clear previous hover text
                axs[0, 1].axvline(x=x, color=current_palette['hover_line_color'], linestyle='--')
                axs[0, 1].text(x, axs[0, 1].get_ylim()[1], f'{x}', color=current_palette['hover_text_color'], verticalalignment='top')
                canvas.draw_idle()

def jump_to_segment():
    global current_segment_index
    try:
        index = int(segment_entry.get())
        if 0 <= index < len(segment_data):
            current_segment_index = index
            bpm_xlim = axs[0, 1].get_xlim()
            plot_segment(current_segment_index, bpm_xlim=bpm_xlim)
            segment_entry.delete(0, tk.END)  # Clear the entry upon successful jump
    except ValueError:
        pass

root.bind('<Key>', on_key)
fig.canvas.mpl_connect('button_press_event', on_bpm_click)
fig.canvas.mpl_connect('motion_notify_event', on_bpm_hover)

default = None
plot_segment(current_segment_index)
default = axs[0, 1].get_xlim()

span1 = SpanSelector(axs[0, 0], on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=current_palette['span_color']))
span2 = SpanSelector(axs[1, 0], on_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=current_palette['span_color']))
span3 = SpanSelector(axs[0, 1], on_bpm_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor=current_palette['span_color']))
button_frame = tk.Frame(root, bg=current_palette['bg'])
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

reset_button = tk.Button(button_frame, text="Reset Zoom", command=reset_zoom, bg=current_palette['button_bg'], fg=current_palette['button_fg'])
reset_button.pack(side=tk.LEFT, padx=10, pady=5)

night_mode_button = tk.Button(button_frame, text="Toggle Night Mode", command=toggle_night_mode, bg=current_palette['button_bg'], fg=current_palette['button_fg'])
night_mode_button.pack(side=tk.RIGHT, padx=10, pady=5)

segment_entry = tk.Entry(button_frame, bg=current_palette['button_bg'], fg=current_palette['button_fg'])
segment_entry.pack(side=tk.LEFT, padx=10, pady=5)
segment_entry.bind('<Return>', lambda event: jump_to_segment())

jump_button = tk.Button(button_frame, text="Jump to Segment", command=jump_to_segment, bg=current_palette['button_bg'], fg=current_palette['button_fg'])
jump_button.pack(side=tk.LEFT, padx=10, pady=5)

def on_closing():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
