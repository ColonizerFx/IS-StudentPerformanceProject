# modules/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.cluster import KMeans


def evaluate_classification(model, X_test, y_test, task="grade_category"):
    predictions = model.predict(X_test)
    print(f"\n--- Naive Bayes Classification Report ({task.upper()}) ---")
    print(classification_report(y_test, predictions))
    if task == "pass_fail":
        labels = ['Fail', 'Pass']
    else:
        labels = ['Grade A', 'Grade B', 'Grade C', 'Grade D']
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title(f'Naive Bayes Confusion Matrix ({task})')
    plt.ylabel('Actual Ground Truth')
    plt.xlabel('Model Predictions')
    plt.tight_layout()
    plt.show()


def plot_elbow_method(X_data, cluster_range):
    inertia = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_data)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(cluster_range, inertia, marker='o', color='darkblue', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Curve Optimization for K-Means')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_clusters(X_raw, cluster_labels):
    df_analyzed = X_raw.copy()
    df_analyzed['Cluster'] = cluster_labels
    cluster_profile = df_analyzed.groupby('Cluster').mean()
    print("\n--- K-Means Student Cluster Analysis Profile ---")
    print("Use these averages to define: High-Performing, Average, or At-Risk groups:")
    print(cluster_profile.round(2))
    return cluster_profile


def compute_silhouette_score(X_data, labels):
    score = silhouette_score(X_data, labels)
    print(f"\n--- Silhouette Score (k={len(set(labels))}): {score:.4f} ---")
    print("(Closer to +1 = well-defined, well-separated clusters)")
    return score


def plot_cluster_scatter(X_data, labels, n_clusters, feature_names=None):
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    plt.figure(figsize=(7, 5))
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        plt.scatter(
            X_data[mask, 0], X_data[mask, 1],
            c=colors[cluster_id % len(colors)],
            label=f'Cluster {cluster_id}',
            alpha=0.6, edgecolors='k', linewidths=0.3
        )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f'K-Means Cluster Scatter Plot (k={n_clusters})')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()
