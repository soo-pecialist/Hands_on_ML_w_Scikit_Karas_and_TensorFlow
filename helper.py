import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class PCA_Sphering(BaseEstimator, TransformerMixin):
    """Feature sclaing using PCA Sphering (centering, rotating, scaling)"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        m, n = X.shape
    
        self.mean_ = np.mean(X, axis=0) # centering
        X_centered = X - self.mean_
        self.U_, self.s_, self.Vt_ = np.linalg.svd(X_centered) # Vt for rotating
        self.D_ = np.zeros((n,n))
        self.D_[:n, :n] = np.diag(np.power(self.s_, -0.5)) # for scaling
        return self

    def transform(self, X, y=None):
        return (X-self.mean_) @ self.Vt_.T @  self.D_


class DBSCAN_with_predict(DBSCAN):
    def __init__(self, eps=0.5, *, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        super().__init__(eps=eps, min_samples=min_samples, metric=metric,
                 metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, 
                 p=p, n_jobs=n_jobs)


    def predict(self, X, y=None, n_neighbors = None):
        if n_neighbors:
            self.n_neighbors = n_neighbors 
        
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.components_, self.labels_[self.core_sample_indices_])
        
        return knn.predict(X)
    
    def predict_anomaly(self, X, y=None, n_neighbors = None):
        if n_neighbors:
            self.n_neighbors = n_neighbors 
        
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.components_, self.labels_[self.core_sample_indices_])

        y_pred = knn.predict(X)
        y_dist, y_closest_idx = knn.kneighbors(X, n_neighbors=1)
        y_pred[y_dist.ravel() > self.eps] = -1

        return y_pred.ravel()