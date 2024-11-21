import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ECGClusterer:
    def __init__(self, data_path, hz, seconds):
        self.data_path = data_path
        self.hz = hz
        self.seconds = seconds

    def load_data(self):
        return np.fromfile(self.data_path, dtype=np.int16)

    def preprocess_data(self, data):
        data = self._normalize_data(data)
        segments = self._segment_data(data)
        features = [self._extract_features(segment) for segment in segments]
        self._rank_features_with_pca(features)
        return np.array(features)

    def kmeans_clustering(self, features, n_clusters):
        scaled_features = self._scale_features(features)
        reduced_features = self._reduce_dimensionality(scaled_features)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(reduced_features)
        return kmeans.labels_

    def _normalize_data(self, data):
        return (data - np.mean(data)) / np.std(data)

    def _segment_data(self, data):
        window_size = self.hz * self.seconds
        return [data[i:i + window_size] for i in range(0, len(data), window_size)]

    def _extract_features(self, segment):
        f_nyquist = self.hz / 2
        vlf_band = (0.003 * f_nyquist, 0.04 * f_nyquist)
        lf_band = (0.04 * f_nyquist, 0.15 * f_nyquist)
        hf_band = (0.15 * f_nyquist, 0.4 * f_nyquist)

        freqs, psd = signal.welch(segment, fs=self.hz, nperseg=1024)
        total_power = np.sum(psd)
        if total_power == 0:
            return []

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

        return [
            total_power, vlf_power, lf_power, hf_power, 
            vlf_ratio, lf_ratio, lf_hf_ratio, 
            spectral_entropy, mean_freq, 
            std_val, 
            max_val, min_val, range_val, rms_val, 
            skewness, mean_val, kurtosis, peak_freq, bandwidth, median_freq, hf_ratio
        ]

    def _rank_features_with_pca(self, features):
        feature_names = [
            "Total Power", "VLF Power", "LF Power", "HF Power", "VLF Ratio", 
            "LF Ratio", "LF/HF Ratio", "Spectral Entropy", 
            "Mean Frequency","Peak Frequency", 
            "Standard Deviation", "Maximum Value", 
            "Minimum Value", "Range", "RMS Value", "Skewness", "Kurtosis",
            "Peak Frequency", "Bandwidth", "Median Frequency", "HF Ratio"
        ]

        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(features)
        pca = PCA()
        pca.fit(standardized_features)
        component_contributions = np.abs(pca.components_)
        explained_variance = pca.explained_variance_ratio_
        feature_importance = np.dot(component_contributions.T, explained_variance)
        feature_ranking = sorted(
            zip(feature_names, feature_importance),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (feature, importance) in enumerate(feature_ranking):
            print(f"{i+1}. {feature}: {importance:.4f}")

    def _scale_features(self, features):
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def _reduce_dimensionality(self, features):
        pca = PCA(n_components=0.95)
        return pca.fit_transform(features)


# Example usage
if __name__ == "__main__":
    data_path = "/home/pavel/py/school/PZS/data/mit-bih-normal-sinus-rhythm-database-1.0.0/18184.dat"
    hz = 128
    seconds = 10

    clusterer = ECGClusterer(data_path, hz, seconds)
    data = clusterer.load_data()
    features = clusterer.preprocess_data(data)
    labels = clusterer.kmeans_clustering(features, n_clusters=3)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    clusterer.fig = fig
    clusterer.axs = axs
    clusterer.labels = labels
    clusterer.features = features

    pca = PCA(n_components=2)

    reduced_features = pca.fit_transform(features)
    scatter = axs[1].scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    fig.colorbar(scatter, ax=axs[1])
    axs[1].set_title('ECG Clusters')
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    plt.show()
