import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import adjusted_rand_score
from sklearn.base import BaseEstimator, ClusterMixin


class KMeansClustering(BaseEstimator, ClusterMixin):
    def __init__(
        self, k: int = 3, max_iter: int = 200, conv_thresh: float = 1e-4
    ) -> None:
        self.k = k
        self.max_iter = max_iter
        self.conv_thresh = conv_thresh
        self.centroids = None
        self.labels_ = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum(np.square(centroids - data_point), axis=1))

    def fit(self, X: np.ndarray | pd.DataFrame) -> 'KMeansClustering':
        X = X.values if isinstance(X, pd.DataFrame) else X

        self.centroids = np.random.uniform(
            low=np.amin(X, axis=0),
            high=np.amax(X, axis=0),
            size=(
                self.k,
                X.shape[1],
            ),
        )

        for _ in range(self.max_iter):
            y = []
            cluster_indices = []
            cluster_centers = []

            for data_point in X:
                distances = self.euclidean_distance(
                    data_point=data_point,
                    centroids=self.centroids,
                )
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)
            self.labels_ = y

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            new_centroids = np.array(cluster_centers)

            if np.max(self.centroids - new_centroids) < 1e-4:
                break

            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:

        X = X.values if isinstance(X, pd.DataFrame) else X

        predictions = []
        for data_point in X:
            distances = self.euclidean_distance(data_point, self.centroids)
            predictions.append(np.argmin(distances))
        return np.array(predictions)

    def fit_predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:

        X = X.values if isinstance(X, pd.DataFrame) else X

        distances = np.array([self.euclidean_distance(x, self.centroids) for x in X])
        return distances

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def score(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame = None
    ) -> float:

        X = X.values if isinstance(X, pd.DataFrame) else X

        distances = self.transform(X)
        min_distances = np.min(distances, axis=1)
        unsupervised_score = -np.sum(np.square(min_distances))

        if y is not None:
            predicted_labels = self.predict(X)
            supervised_score = adjusted_rand_score(y, predicted_labels)
            return supervised_score

        return unsupervised_score
