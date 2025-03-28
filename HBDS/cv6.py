# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def pipeline(path, k):
    if 0:
        data = pd.read_csv(path, header=None, names=["V1", "V2", "Cluster"], sep='\t')
    else:
        data = pd.read_csv(path)

    print(data)

    data.info()
    data.head()

    print("Dataset shape:", data.shape)

    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors = nearest_neighbors.fit(data[["V1", "V2"]])
    distances, indices = neighbors.kneighbors(data[["V1", "V2"]])

    # Sort the distances to identify the "elbow" point
    distances = np.sort(distances[:, -1])

    #Plot the k-distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-distance')
    plt.title('k-Distance Plot for DBSCAN')
    plt.grid(True)
    plt.show()


    def dbscan_evaluation(data, eps_min, eps_max, eps_step, k):
        """
        Evaluate DBSCAN clustering for a range of eps values.

        Parameters:
            data (DataFrame): Dataset with 'x' and 'y' columns.
            eps_min (float): Minimum value of eps.
            eps_max (float): Maximum value of eps.
            eps_step (float): Step size for changing eps.
            k (int): Min_samples for DBSCAN.

        Returns:
            None: Plots three charts for Silhouette, Calinski-Harabasz, and Davies-Bouldin criteria.
        """
        eps_values = np.arange(eps_min, eps_max, eps_step)
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []

        for eps in eps_values:
            # Apply DBSCAN with the current eps value
            dbscan = DBSCAN(eps=eps, min_samples=k).fit(data)
            labels = dbscan.labels_

            # Compute clustering metrics (handle cases with only one cluster)
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(data, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(data, labels))
                davies_bouldin_scores.append(davies_bouldin_score(data, labels))
            else:
                silhouette_scores.append(np.nan)
                calinski_harabasz_scores.append(np.nan)
                davies_bouldin_scores.append(np.nan)

        # Plotting the metrics versus eps
        plt.figure(figsize=(10, 15))

        # Silhouette Score plot
        plt.subplot(3, 1, 1)
        plt.plot(eps_values, silhouette_scores, marker='o', linestyle='-', color='blue')
        plt.title('Silhouette Score vs. eps')
        plt.xlabel('eps')
        plt.ylabel('Silhouette Score')
        plt.grid(True)

        # Calinski-Harabasz Index plot
        plt.subplot(3, 1, 2)
        plt.plot(eps_values, calinski_harabasz_scores, marker='o', linestyle='-', color='green')
        plt.title('Calinski-Harabasz Index vs. eps')
        plt.xlabel('eps')
        plt.ylabel('Calinski-Harabasz Index')
        plt.grid(True)

        # Davies-Bouldin Index plot
        plt.subplot(3, 1, 3)
        plt.plot(eps_values, davies_bouldin_scores, marker='o', linestyle='-', color='red')
        plt.title('Davies-Bouldin Index vs. eps')
        plt.xlabel('eps')
        plt.ylabel('Davies-Bouldin Index')
        plt.grid(True)

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    #dbscan_evaluation(data, eps_min=0.5, eps_max=3, eps_step=0.01, k = k)

    eps_candidates = np.linspace(1.4, 1.5, 10)
    

    for eps_value in eps_candidates:

        dbscan = DBSCAN(eps=eps_value, min_samples=k).fit(data)
        dbscan_labels = dbscan.labels_

        # Visualize DBSCAN clustering results
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data["V1"], y=data["V2"], hue=dbscan_labels, palette="tab10", s=50, edgecolor="k")
        plt.title(f"DBSCAN Clustering Results, EPS: {eps_value}")
        plt.xlabel("V1")
        plt.ylabel("V2")
        plt.legend(title="Cluster", loc="upper right", bbox_to_anchor=(1.15, 1))
        plt.show()

def DBSCN(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    dbscan_labels = dbscan.labels_

    # Visualize DBSCAN clustering results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data["V1"], y=data["V2"], hue=dbscan_labels, palette="tab10", s=50, edgecolor="k")
    plt.title(f"DBSCAN Clustering Results, EPS: {eps}")
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.legend(title="Cluster", loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.show()


if __name__ == "__main__":
    #pipeline("HBDS/Datasets/Compound.csv", 3)# 1.45
    #pipeline("HBDS/Datasets/aggregation.csv", 8) #1.55
    #pipeline("HBDS/Datasets/jain.txt", 2) #2.63
    #pipeline("HBDS/Datasets/spiral.txt", 2) #2.63

    data = pd.read_csv("HBDS/Datasets/Compound.csv")
    DBSCN(data, 1.4, 3)
    data = pd.read_csv("HBDS/Datasets/aggregation.csv")
    DBSCN(data, 1.55, 8)
    data = pd.read_csv("HBDS/Datasets/jain.txt", header=None, names=["V1", "V2", "Cluster"], sep='\t')
    DBSCN(data, 2.63, 2)
    data = pd.read_csv("HBDS/Datasets/spiral.txt", header=None, names=["V1", "V2", "Cluster"], sep='\t')
    DBSCN(data, 2.63, 2)

