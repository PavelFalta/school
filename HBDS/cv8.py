import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt


path_data = 'HBDS/Datasets/alzh_data_filtered.csv'

df_data = pd.read_csv(path_data)



print(df_data['Class'].value_counts())


Y = df_data['Class']
df_tdata = df_data.drop(columns=['Class'])
df_data = df_tdata.T

print(df_data.shape)


def evaluate_clustering(data, k_min, k_max, algorithm="kmeans"):
    ch_scores = []
    silhouette_scores = []
    db_scores = []
    k_range = range(k_min, k_max + 1)

    for k in k_range:
        if algorithm == "kmeans":
            model = KMeans(n_clusters=k, random_state=42)
        elif algorithm == "kmedoids":
            model = KMedoids(n_clusters=k, random_state=42)
        else:
            raise ValueError("Algorithm must be 'kmeans' or 'kmedoids'.")

        labels = model.fit_predict(data)

        ch_scores.append(calinski_harabasz_score(data, labels))
        silhouette_scores.append(silhouette_score(data, labels))
        db_scores.append(davies_bouldin_score(data, labels))

    return k_range, ch_scores, silhouette_scores, db_scores


def plot_metrics(k_range, ch_scores, silhouette_scores, db_scores, algorithm_name):
    plt.figure(figsize=(15, 5))

    
    plt.subplot(1, 3, 1)
    plt.plot(k_range, ch_scores, marker='o')
    plt.title(f'{algorithm_name} - Calinski-Harabasz Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title(f'{algorithm_name} - Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    
    plt.subplot(1, 3, 3)
    plt.plot(k_range, db_scores, marker='o')
    plt.title(f'{algorithm_name} - Davies-Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()


def run_clustering(data, k_min, k_max):
    
    k_range, ch_scores, silhouette_scores, db_scores = evaluate_clustering(data, k_min, k_max, algorithm="kmeans")
    plot_metrics(k_range, ch_scores, silhouette_scores, db_scores, "KMeans")

    
    k_range, ch_scores, silhouette_scores, db_scores = evaluate_clustering(data, k_min, k_max, algorithm="kmedoids")
    plot_metrics(k_range, ch_scores, silhouette_scores, db_scores, "KMedoids")

# run_clustering(df_data, k_min=2, k_max=10)

def cluster_genes(data, n_clusters=2):
    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    gene_labels = clustering.fit_predict(data)  
    return gene_labels

gene_clusters = cluster_genes(df_data)


def split_data_by_clusters(data, gene_clusters, cluster_id):
    
    cluster_columns = np.where(gene_clusters == cluster_id)[0]
    
    return data.iloc[:, cluster_columns]


data_cluster_1 = split_data_by_clusters(df_tdata, gene_clusters, cluster_id=0)
data_cluster_2 = split_data_by_clusters(df_tdata, gene_clusters, cluster_id=1)

print(data_cluster_1.shape)
print(data_cluster_2.shape)


def train_random_forest(X, y):
    model = RandomForestClassifier(
      n_estimators=100,    
      bootstrap=True,      
      max_depth=10,        
      max_features='sqrt', 
      min_samples_leaf=1,  
      min_samples_split=10,
      random_state=42)

    model.fit(X, y)
    return model


X_train, X_test, y_train, y_test = train_test_split(df_tdata, Y, test_size=0.3, random_state=42)


X_train_cluster_1 = split_data_by_clusters(X_train, gene_clusters, cluster_id=0)
X_train_cluster_2 = split_data_by_clusters(X_train, gene_clusters, cluster_id=1)

X_test_cluster_1 = split_data_by_clusters(X_test, gene_clusters, cluster_id=0)
X_test_cluster_2 = split_data_by_clusters(X_test, gene_clusters, cluster_id=1)


rf_model_1 = train_random_forest(X_train_cluster_1, y_train)
rf_model_2 = train_random_forest(X_train_cluster_2, y_train)


def train_stacking_model(X_train_1, X_train_2, y_train):
    
    preds_train_1 = rf_model_1.predict_proba(X_train_1)[:, 1]
    preds_train_2 = rf_model_2.predict_proba(X_train_2)[:, 1]

    
    meta_features_train = np.column_stack((preds_train_1, preds_train_2))

    
    meta_model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    meta_model.fit(meta_features_train, y_train)

    return meta_model

def test_stacking_model(meta_model, rf_model_1, rf_model_2, X_test_cluster_1, X_test_cluster_2, y_test):
    """
    Test the trained stacking model on test datasets and display evaluation metrics.

    Parameters:
        meta_model: Trained logistic regression metamodel.
        rf_model_1: Trained Random Forest classifier for cluster 1.
        rf_model_2: Trained Random Forest classifier for cluster 2.
        X_test_cluster_1: Test data corresponding to cluster 1.
        X_test_cluster_2: Test data corresponding to cluster 2.
        y_test: True labels for the test data.
    """
    
    preds_test_1 = rf_model_1.predict_proba(X_test_cluster_1)[:, 1]
    preds_test_2 = rf_model_2.predict_proba(X_test_cluster_2)[:, 1]

    
    meta_features_test = np.column_stack((preds_test_1, preds_test_2))

    
    predictions = meta_model.predict(meta_features_test)

    
    rf_cf = classification_report(y_test, predictions)
    print(f"Classification Report:\n{rf_cf}")

    
    conf_matrix = confusion_matrix(y_test, predictions)

    
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Stacking Classifier')
    plt.show()



meta_model = train_stacking_model(X_train_cluster_1, X_train_cluster_2, y_train)

test_stacking_model(meta_model, rf_model_1, rf_model_2, X_test_cluster_1, X_test_cluster_2, y_test)


def cluster_genes(data, n_clusters=2):
    clustering = KMedoids(n_clusters=n_clusters, random_state=42)
    gene_labels = clustering.fit_predict(data)  
    return gene_labels

gene_clusters = cluster_genes(df_data)

data_cluster_1 = split_data_by_clusters(df_tdata, gene_clusters, cluster_id=0)
data_cluster_2 = split_data_by_clusters(df_tdata, gene_clusters, cluster_id=1)


X_train, X_test, y_train, y_test = train_test_split(df_tdata, Y, test_size=0.3, random_state=42)


X_train_cluster_1 = split_data_by_clusters(X_train, gene_clusters, cluster_id=0)
X_train_cluster_2 = split_data_by_clusters(X_train, gene_clusters, cluster_id=1)

X_test_cluster_1 = split_data_by_clusters(X_test, gene_clusters, cluster_id=0)
X_test_cluster_2 = split_data_by_clusters(X_test, gene_clusters, cluster_id=1)


rf_model_1 = train_random_forest(X_train_cluster_1, y_train)
rf_model_2 = train_random_forest(X_train_cluster_2, y_train)


meta_model = train_stacking_model(X_train_cluster_1, X_train_cluster_2, y_train)

test_stacking_model(meta_model, rf_model_1, rf_model_2, X_test_cluster_1, X_test_cluster_2, y_test)