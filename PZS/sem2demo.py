import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class ECGClusterer:
    def __init__(self, data_path, hz, seconds):
        self.data_path = data_path
        self.hz = hz
        self.seconds = seconds

    def load_data(self):
        # načtení dat ze souboru
        return np.fromfile(self.data_path, dtype=np.int16)

    def preprocess_data(self, data):
        # předzpracování dat
        data = self._normalize_data(data)
        segments = self._segment_data(data)
        features = [self._extract_features(segment) for segment in segments]
        #self._rank_features_with_pca(features)
        return np.array(features)

    def kmeans_clustering(self, features, n_clusters):
        # k-means clustering
        scaled_features = self._scale_features(features)
        reduced_features = self._reduce_dimensionality(scaled_features)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(reduced_features)
        self.labels = kmeans.labels_
        return reduced_features, self.labels

    def _normalize_data(self, data):
        # normalizace dat
        return (data - np.mean(data)) / np.std(data)

    def _segment_data(self, data):
        # segmentace dat
        window_size = self.hz * self.seconds
        return [data[i:i + window_size] for i in range(0, len(data), window_size)]

    def _extract_features(self, segment):
        # extrakce příznaků ze segmentu
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
                skewness
        ]

    def _rank_features_with_pca(self, features):
        # hodnocení příznaků pomocí pca
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
        # škálování příznaků
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def _reduce_dimensionality(self, features):
        # redukce dimenzionality pomocí pca
        pca = PCA(n_components=0.95)
        return pca.fit_transform(features)

    def _highlight_good_cluster(self, reduced_features):
        # zvýraznění dobrého clusteru
        cluster_centers, cluster_sizes, middle_point = self._calculate_cluster_properties(reduced_features)
        good_cluster_label = self._determine_good_cluster(cluster_centers, cluster_sizes, middle_point)
        self._draw_cluster_boundaries(reduced_features, good_cluster_label)

    def _calculate_cluster_properties(self, reduced_features):
        # výpočet vlastností clusteru
        cluster_centers = np.array([reduced_features[self.labels == label].mean(axis=0) for label in np.unique(self.labels)])
        cluster_sizes = np.array([np.sum(self.labels == label) for label in np.unique(self.labels)])
        middle_point = np.mean(reduced_features, axis=0)
        return cluster_centers, cluster_sizes, middle_point

    def _determine_good_cluster(self, cluster_centers, cluster_sizes, middle_point):
        # určení dobrého clusteru
        distances_to_middle = np.linalg.norm(cluster_centers - middle_point, axis=1)
        normalized_sizes = cluster_sizes / np.max(cluster_sizes)
        normalized_distances = distances_to_middle / np.max(distances_to_middle)
        scores = normalized_sizes - normalized_distances
        good_cluster_index = np.argmax(scores)
        return np.unique(self.labels)[good_cluster_index]

    def _draw_cluster_boundaries(self, reduced_features, good_cluster_label):
        # vykreslení hranic clusteru
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
                plt.plot(moved_points[simplex, 0], moved_points[simplex, 1], 'r--', lw=2, label='Predicted Good Signal' if not i else None)
        plt.legend()

if __name__ == "__main__":
    data_path = "/home/pavel/py/school/PZS/data/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/122001/122001_ECG.dat"
    hz = 1000
    seconds = 10

    print("načítání dat...")
    clusterer = ECGClusterer(data_path, hz, seconds)
    data = clusterer.load_data()
    print("data načtena.")

    print("předzpracování dat...")
    features = clusterer.preprocess_data(data)
    print("předzpracování dokončeno.")

    print("provádění k-means clusteringu...")
    reduced_features, labels = clusterer.kmeans_clustering(features, n_clusters=3)
    print("k-means clustering dokončen.")

    print("redukce dimenzionality pomocí pca...")
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    print("redukce dimenzionality dokončena.")

    print("vykreslování grafu...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('ECG Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    print("zvýraznění dobrého clusteru...")
    clusterer._highlight_good_cluster(reduced_features)
    
    biggest_cluster_label = np.argmax(np.bincount(labels))
    biggest_cluster_size = np.sum(labels == biggest_cluster_label)
    print(f"velikost největšího clusteru: {biggest_cluster_size}")

    # počítání statistik
    good_cluster_label = clusterer._determine_good_cluster(*clusterer._calculate_cluster_properties(reduced_features))
    good_cluster_points = reduced_features[labels == good_cluster_label]
    centroid = np.mean(good_cluster_points, axis=0)
    distances_from_centroid = np.linalg.norm(good_cluster_points - centroid, axis=1)
    max_distance = np.percentile(distances_from_centroid, 50)
    anomalies = np.sum(distances_from_centroid >= max_distance)
    total_segments = len(reduced_features)
    good_segments = total_segments - anomalies
    anomaly_percentage = (anomalies / total_segments) * 100
    good_percentage = (good_segments / total_segments) * 100

    print(f"dobré segmenty: {good_segments} ({good_percentage:.2f}%)")
    print(f"anomálie: {anomalies} ({anomaly_percentage:.2f}%)")

    plt.show()
    print("hotovo.")
