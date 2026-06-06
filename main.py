# main.py
import config
from modules.data_loader import load_dataset
from modules.preprocessing import preprocess_data
from modules.models import train_naive_bayes, run_kmeans
from modules.evaluation import (
    evaluate_classification, plot_elbow_method, analyze_clusters,
    compute_silhouette_score, plot_cluster_scatter
)
from sklearn.preprocessing import StandardScaler

def main():
    print("🚀 Running Student Performance Intelligent System Frame in VS Code...")
    
    # 1. Pipeline Ingestion
    df = load_dataset(config.DATA_PATH)
    if df is None:
        return
        
    # 2. Preprocessing & Data Splits
    X_train_scaled, X_test_scaled, y_train, y_test, X_raw = preprocess_data(
        df, 
        task=config.CLASSIFICATION_TASK,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # 3. Supervised Path: Naive Bayes
    # Experiment with different var_smoothing values (Gaussian NB hyperparameter tuning)
    print(f"\nTraining Naive Bayes Classifier for task: '{config.CLASSIFICATION_TASK}'...")
    print(f"Testing var_smoothing values: {config.NB_VAR_SMOOTHING}")
    best_nb_model = None
    for smoothing in config.NB_VAR_SMOOTHING:
        print(f"\n[var_smoothing = {smoothing}]")
        nb_model = train_naive_bayes(X_train_scaled, y_train, var_smoothing=smoothing)
        evaluate_classification(nb_model, X_test_scaled, y_test, task=config.CLASSIFICATION_TASK)
        best_nb_model = nb_model  # keep last model for reference
    
    # 4. Unsupervised Path: K-Means Optimal Target Clustering
    print("\nEvaluating Cluster Counts via Elbow Plot...")
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_raw)
    plot_elbow_method(X_full_scaled, config.KMEANS_CLUSTERS)
    
    # Fit the 3 explicit cluster profiles reque