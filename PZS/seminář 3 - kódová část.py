from data.loader import Loader
import numpy as np
from plotly import express as px
from scipy.signal import butter, filtfilt, welch
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def avg_filter(signal, window_size):
    #vypocet kumulativniho souctu signalu
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    #vypocet prumerneho filtru
    filtered_signal = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return filtered_signal

def lowpass_filter(signal, cutoff_freq, fs):
    #vypocet nyquistovy frekvence
    nyquist = 0.5 * fs
    #normalizace omezovaci frekvence
    normal_cutoff = cutoff_freq / nyquist
    #navrh dolniho propustneho filtru
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    #aplikace filtru na signal
    filtered_signal = filtfilt(b, a, signal)
    return np.array(filtered_signal)

def extract_features(signal):
    #vypocet statistickych vlastnosti signalu
    signal_std = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    peak_to_peak = max_val - min_val

    #vypocet vlastnosti ve frekvencni domene pomoci Welchovy metody
    freqs, psd = welch(signal, fs=8000, nperseg=1024)
    max_psd = np.max(psd)
    mean_psd = np.mean(psd)
    median_psd = np.median(psd)
    std_psd = np.std(psd)
    freq_with_max_psd = freqs[np.argmax(psd)]

    return [signal_std, peak_to_peak, max_psd, mean_psd, median_psd, std_psd, freq_with_max_psd]

def perform_pca(data, n_components=2):
    #provedeni PCA na datech
    pca_model = PCA(n_components=n_components)
    transformed_data = pca_model.fit_transform(data)
    return transformed_data

def load_and_preprocess_data(loader_path, fs=8000):
    #nacteni dat pomoci Loaderu
    loader = Loader(loader_path)
    loader.recursive_search()
    records = loader.records

    combined_data = []
    for rec in records:
        #ziskani originalniho signalu
        original_signal = np.array(rec.data[3000:])
        #aplikace dolniho propustneho filtru
        filtered_signal = lowpass_filter(original_signal, cutoff_freq=100, fs=fs)
        #aplikace prumerneho filtru
        processed_signal = avg_filter(filtered_signal, 1000)
        combined_data.append((rec.diagnosis, original_signal, filtered_signal, processed_signal))

    #serazeni dat podle diagnozy
    combined_data.sort(key=lambda x: x[0])
    return combined_data

def analyze_and_plot(combined_data):
    features = []
    diagnoses = []
    for diagnosis, original_signal, raw_signal, processed_signal in combined_data:
        #extrakce vlastnosti zpracovaneho signalu
        features.append(extract_features(processed_signal))
        #normalizace diagnozy
        normalized_diag = diagnosis if diagnosis != "hyperkineti dysphonia" else "hyperkinetic dysphonia"
        diagnoses.append(normalized_diag)

    #provedeni PCA na extrahovanych vlastnostech
    pca_data = perform_pca(features)

    #vypocet stredu PCA transformovanych dat
    center = np.mean(pca_data, axis=0)

    #vypocet euklidovskych vzdalenosti od kazdeho bodu ke stredu
    distances = np.linalg.norm(pca_data - center, axis=1)

    #urceni nejblizsich 50% dat
    num_closest = int(len(distances) / 2)
    closest_indices = np.argsort(distances)[:num_closest]
    closest_labels = np.array(diagnoses)[closest_indices]

    #pocet vyskytu v nejblizsich 50% a celkove
    unique_closest, counts_closest = np.unique(closest_labels, return_counts=True)
    print("Counts in the closest 50%:")
    for label, count in zip(unique_closest, counts_closest):
        print(f"{label}: {count}")
    
    unique_all, counts_all = np.unique(diagnoses, return_counts=True)
    print("\nCounts in the whole dataset:")
    for label, count in zip(unique_all, counts_all):
        print(f"{label}: {count}")
    overall_counts = {label: count for label, count in zip(unique_all, counts_all)}
    percentages_closest = np.array([
        count / overall_counts[label] * 100 for label, count in zip(unique_closest, counts_closest)
    ])

    #identifikace outlieru jako bodu, ktere nejsou v nejblizsich 50%
    all_indices = np.arange(len(distances))
    outlier_indices = np.setdiff1d(all_indices, closest_indices)

    #vykresleni PCA scatter plotu
    fig_scatter = px.scatter(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        color=diagnoses,
        title="PCA"
    )
    fig_scatter.add_scatter(
        x=[center[0]],
        y=[center[1]],
        mode='markers',
        marker=dict(size=10, color='black', symbol='x'),
        name="Střed"
    )
    fig_scatter.add_scatter(
        x=pca_data[outlier_indices, 0],
        y=pca_data[outlier_indices, 1],
        mode='markers',
        marker=dict(size=10, color='red',symbol='circle-open'),
        name="Outliers"
    )
    fig_scatter.show()

    #vykresleni bar chartu pro procento kazdeho labelu v nejblizsich 50% dat
    fig_bar = px.bar(
        x=unique_closest,
        y=percentages_closest,
        title="Procento jednotlivých diagnóz v nejbližších 50% od středu",
        labels={'x': 'Diagnóza', 'y': 'Procento zastoupení (%)'}
    )
    fig_bar.add_bar(
        x=unique_closest,
        y=100 - percentages_closest,
        opacity=0.5,
        name="Outliers"
    )
    fig_bar.show()

    return outlier_indices

def display_outlier_signals_separately(combined_data, outlier_indices):
    if len(outlier_indices) == 0:
        print("No outliers to display.")
        return

    #zamichani indexu pro nahodne zobrazeni outlieru
    np.random.shuffle(outlier_indices)

    for idx in outlier_indices:
        diagnosis, original_signal, raw_signal, processed_signal = combined_data[idx]

        fig = go.Figure()

        #vykresleni originalniho signalu modre
        fig.add_trace(go.Scatter(
            x=list(range(len(original_signal))),
            y=original_signal,
            mode="lines",
            name="Původní",
            line=dict(color="blue")
        ))

        #vykresleni zpracovaneho signalu cervene
        fig.add_trace(go.Scatter(
            x=list(range(len(raw_signal))),
            y=raw_signal,
            mode="lines",
            name="Filtrovaný dolní propustí",
            line=dict(color="red")
        ))
        fig.update_layout(
            title=f"Outlier Index {idx} - Diagnóza: {diagnosis}",
            xaxis_title="Sample Index",
            yaxis_title="Signal Amplitude"
        )
        fig.show()

        fig = go.Figure()
        #vykresleni zpracovaneho signalu modre
        fig.add_trace(go.Scatter(
            x=list(range(len(raw_signal))),
            y=raw_signal,
            mode="lines",
            name="Filtrovaný dolní propustí",
            line=dict(color="blue")
        ))
        #vykresleni prumerneho filtru cervene
        fig.add_trace(go.Scatter(
            x=list(range(len(processed_signal))),
            y=processed_signal,
            mode="lines",
            name="Průměrový filtr",
            line=dict(color="red")
        ))
        fig.update_layout(
            title=f"Outlier Index {idx} - Diagnóza: {diagnosis}",
            xaxis_title="Sample Index",
            yaxis_title="Signal Amplitude"
        )
        fig.show()
        input("Press Enter to display the next outlier...")

if __name__ == "__main__":
    FS = 8000  #puvodni vzorkovaci frekvence
    #nacteni dat a vypocet jak surovych, tak zpracovanych signalu
    combined_data = load_and_preprocess_data("data/voice_data", fs=FS)
    #analyza vlastnosti, provedeni PCA a vykresleni PCA a bar chartu
    outlier_indices = analyze_and_plot(combined_data)
    #pro kazdy outlier (cervene markery v PCA plotu) zobrazeni surovych a zpracovanych signalu samostatne,
    display_outlier_signals_separately(combined_data, outlier_indices)
