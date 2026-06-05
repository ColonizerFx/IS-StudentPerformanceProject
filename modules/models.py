# modules/models.py
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

def train_naive_bayes(X_train, y_train, var_smoothing=1e-9):
    """Instantiates and trains a Gaussian Naive Bayes Classifier."""
    nb = GaussianNB(var_smoothing=var_smoothing)
    nb.fit(X_train, y_train)
    return nb

def run_kmeans(X_data, n_clusters=3, random_state=42):
    """Runs a standard K-Means clustering pass on data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_data)
    return kmeans, labels