# modules/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans

def evaluate_classification(model, X_test, y_test, task="grade_category"):
    """Prints classification performance reports and plots a confusion matrix."""
    predictions = model.predict(X_test)
    print(f"\n--- 📊 Naive Bayes Classification Report ({task.upper()}) ---")
    print(classification_report(y_test, predictions))
    
    # Setup correct display tags
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
    """Generates the elbow plot to evaluate performance variants of K-Means."""
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
    """Profiles the resulting clusters to match academic presentation targets."""
    df_analyzed = X_raw.copy()
    df_analyzed['Cluster'] = cluster_labels
    
    cluster_profile = df_analyzed.groupby('Cluster').mean()
    print("\n--- 👥 K-Means Student Cluster Analysis Profile ---")
    print("Use these averages to define: High-Performing, Average, or At-Risk groups:")
    print(cluster_profile.round(2))
    return cluster_profile