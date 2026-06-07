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
    print("Running Student Performance Intelligent System...")

    df = load_dataset(config.DATA_PATH)
    if df is None:
        return

    X_train_scaled, X_test_scaled, y_train, y_test, X_raw = preprocess_data(
        df,
        task=config.CLASSIFICATION_TASK,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    # Naive Bayes - test different var_smoothing values
    print(f"\nTraining Naive Bayes for task: '{config.CLASSIFICATION_TASK}'...")
    print(f"var_smoothing values: {config.NB_VAR_SMOOTHING}")
    best_nb_model = None
    for smoothing in config.NB_VAR_SMOOTHING:
        print(f"\n[var_smoothing = {smoothing}]")
        nb_model = train_naive_bayes(X_train_scaled, y_train, var_smoothing=smoothing)
        evaluate_classification(nb_model, X_test_scaled, y_test, task=config.CLASSIFICATION_TASK)
        best_nb_model = nb_model

    # K-Means - find optimal k using elbow method
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_raw)
    plot_elbow_method(X_full_scaled, config.KMEANS_CLUSTERS)

    print("\nRunning K-Means with k=3...")
    final_kmeans, final_cluster_labels = run_kmeans(X_full_scaled, n_clusters=3, random_state=config.RANDOM_STATE)

    analyze_clusters(X_raw, final_cluster_labels)
    compute_silhouette_score(X_full_scaled, final_cluster_labels)

    feature_names = list(X_raw.columns[:2])
    plot_cluster_scatter(X_full_scaled, final_cluster_labels, n_clusters=3, feature_names=feature_names)

    print("\nDone.")

if __name__ == "__main__":
    main()