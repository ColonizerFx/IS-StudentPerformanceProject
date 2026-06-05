# main.py
import config
from modules.data_loader import load_dataset
from modules.preprocessing import preprocess_data
from modules.models import train_naive_bayes, run_kmeans
from modules.evaluation import evaluate_classification, plot_elbow_method, analyze_clusters
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
    print(f"\nTraining Naive Bayes Classifier for task: '{config.CLASSIFICATION_TASK}'...")
    nb_model = train_naive_bayes(X_train_scaled, y_train, var_smoothing=config.NB_VAR_SMOOTHING[0])
    evaluate_classification(nb_model, X_test_scaled, y_test, task=config.CLASSIFICATION_TASK)
    
    # 4. Unsupervised Path: K-Means Optimal Target Clustering
    print("\nEvaluating Cluster Counts via Elbow Plot...")
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_raw)
    plot_elbow_method(X_full_scaled, config.KMEANS_CLUSTERS)
    
    # Fit the 3 explicit cluster profiles requested by your brief
    print("\nGrouping students into 3 target profiles...")
    _, final_cluster_labels = run_kmeans(X_full_scaled, n_clusters=3, random_state=config.RANDOM_STATE)
    
    # Analyze clusters to determine which group is high-performing vs at-risk
    analyze_clusters(X_raw, final_cluster_labels)
    
    print("\n🏁 Process complete!")

if __name__ == "__main__":
    main()