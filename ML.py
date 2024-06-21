import numpy as np
from typing import Literal

class KMeans:
    """
    K-means clustering algorithm implementation.

    Attributes:
        clusters (int): Number of clusters.
        centroids (np.ndarray): Cluster centroids.
        max_iter (int): Maximum number of iterations.
        distance_type (str): Type of distance measure to use.
        assignments (np.ndarray): Cluster assignments for each data point.
    """

    def __init__(
        self,
        k: int,
        max_iter: int = 100,
        distance_type: Literal["euclidean", "manhattan", "cosine"] = "euclidean",
    ):
        """
        Initialize the KMeans instance.

        Args:
            k (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            distance_type (str): Type of distance measure to use.
        """
        if k < 2:
            raise ValueError("Number of clusters must be at least 2")
        if max_iter < 1:
            raise ValueError("Maximum iterations must be at least 1")
        if distance_type not in ["euclidean", "manhattan", "cosine"]:
            raise ValueError("Unsupported distance type")

        self.clusters = k
        self.centroids = None
        self.max_iter = max_iter
        self.distance_type = distance_type
        self.assignments = None

    def _initialize_centroids(self, X: np.ndarray) -> None:
        """
        Initialize cluster centroids randomly within the data range.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        """
        self.centroids = np.random.uniform(
            low=np.min(X, axis=0),
            high=np.max(X, axis=0),
            size=(self.clusters, X.shape[1]),
        )

    def _distance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate distance between data points and centroids.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Distance matrix of shape (n_samples, n_clusters).
        """
        if self.distance_type == "manhattan":
            return np.sum(
                np.abs(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]),
                axis=2,
            )
        elif self.distance_type == "euclidean":
            return np.sqrt(
                np.sum(
                    np.power(
                        (X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]),
                        2,
                    ),
                    axis=2,
                )
            )
        elif self.distance_type == "cosine":
            X_normalized = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
            centroids_normalized = self.centroids / np.linalg.norm(
                self.centroids, axis=1
            )[:, np.newaxis]
            cosine_similarity = np.dot(X_normalized, centroids_normalized.T)
            return 1 - cosine_similarity

    def _update_centroids(self, X: np.ndarray) -> bool:
        """
        Update cluster centroids based on current assignments.
        Handle empty clusters by reinitializing them.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            bool: True if centroids changed, False otherwise.
        """
        distance = self._distance(X)
        self.assignments = np.argmin(distance, axis=1)

        new_centroids = np.zeros_like(self.centroids)
        changed = False

        for i in range(self.clusters):
            cluster_points = X[self.assignments == i]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
                if not np.allclose(new_centroid, self.centroids[i]):
                    changed = True
                new_centroids[i] = new_centroid
            else:
                # Reinitialize empty cluster
                new_centroids[i] = X[np.random.choice(X.shape[0])]
                changed = True

        self.centroids = new_centroids
        return changed

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit the K-means model to the input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            self: The fitted KMeans instance.
        """
        if X.shape[0] < self.clusters:
            raise ValueError(
                "Number of samples must be at least equal to the number of clusters"
            )

        self._initialize_centroids(X)

        for _ in range(self.max_iter):
            if not self._update_centroids(X):
                break

        return self

    def calculate_sse(self, X: np.ndarray) -> float:
        """
        Calculate the Sum of Squared Errors (SSE) for the clustering.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            float: The SSE value.
        """
        distances = self._distance(X)
        sse = np.sum(np.min(distances**2, axis=1))
        return sse / X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        return np.argmin(self._distance(X), axis=1)

class KMeansPlusPlus(KMeans):
    """
    K-means++ clustering algorithm implementation.
    Inherits from KMeans and only changes the initialization method.
    """

    def _initialize_centroids(self, X):
        """
        Initialize cluster centroids using the K-means++ method.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.clusters, n_features))
        
        # Randomly choose the first centroid
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose the remaining centroids
        for i in range(1, self.clusters):
            # Compute distances from points to the centroids
            distances = np.min(np.sum((X[:, np.newaxis] - centroids[:i])**2, axis=2), axis=1)
            
            # Choose the next centroid with probability proportional to distance squared
            probs = distances / distances.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            ind = np.searchsorted(cumprobs, r)
            centroids[i] = X[ind]
        
        self.centroids = centroids

class PCATransformer:
    def __init__(self, explained_variance_ratio=None, n_components=None):
        """
        Initializes the PCATransformer class.
        
        Args:
          explained_variance_ratio (float, optional): The desired explained variance ratio. Defaults to 0.95.
          n_components (int, optional): The desired number of principal components to retain. Defaults to None.
        
        Raises:
          ValueError: If both explained_variance_ratio and n_components are provided.
        """
        self.explained_variance_ratio = explained_variance_ratio
        self.n_components = n_components
        
        if explained_variance_ratio is not None and n_components is not None:
            raise ValueError("Only one of explained_variance_ratio or n_components can be specified.")
        
        self.mean_ = None
        self.std_ = None
        self.covariance_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.principal_components_ = None

    def _validate_input(self, X):
        """
        Validates the input data X.

        Args:
            X (np.ndarray): The input data.

        Raises:
            ValueError: If the input data is not a 2D NumPy array or contains NaN or infinite values.
        """
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array.")

        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Input data contains NaN or infinite values.")

    def _center_and_scale(self, X):
        """
        Centers and scales the input data.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The centered and scaled data.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        # Handle features with zero variance
        self.std_[self.std_ == 0] = 1e-6

        X_centered = (X - self.mean_) / self.std_
        return X_centered

    def _compute_covariance_matrix(self, X):
        """
        Computes the covariance matrix of the input data.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The covariance matrix of shape (n_features, n_features).
        """
        n_samples, n_features = X.shape
        X_centered = self._center_and_scale(X)
        self.covariance_matrix_ = X_centered.T @ X_centered / n_samples
        return self.covariance_matrix_

    def _compute_eigen(self):
        """
        Computes the eigenvalues and eigenvectors of the covariance matrix.
        """
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eigh(self.covariance_matrix_)

        # Sort the eigenvectors in descending order of eigenvalues
        sorted_indices = np.argsort(-abs(self.eigenvalues_))
        self.eigenvalues_ = self.eigenvalues_[sorted_indices]
        self.eigenvectors_ = self.eigenvectors_[:, sorted_indices]

    def _determine_k(self):
        """
        Determines the number of principal components to retain based on the explained variance ratio.
        """
        total_variance = np.sum(self.eigenvalues_)
        explained_variance_ratio = np.cumsum(self.eigenvalues_) / total_variance
        self.n_components = np.argmax(explained_variance_ratio >= self.explained_variance_ratio) + 1

    def fit(self, X):
        """
        Fits the PCA transformer to the input data X.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            self: The fitted PCATransformer instance.
        """
        self._validate_input(X)
        self._compute_covariance_matrix(X)
        self._compute_eigen()
        
        if self.n_components is None:
            self._determine_k()


        # Get the top K principal components
        self.principal_components_ = self.eigenvectors_[:, :self.n_components]

        return self

    def transform(self, X):
        """
        Applies the PCA transformation to the input data X.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data of shape (n_samples, k).
        """
        self._validate_input(X)

        if self.mean_ is None or self.std_ is None or self.principal_components_ is None:
            raise ValueError("PCATransformer must be fitted before transforming data.")

        # Center and scale the data
        X_centered = np.subtract(X, self.mean_)
        X_normalized = np.divide(X, self.std_)

        # Project the data onto the principal component space
        X_transformed = X_centered @ self.principal_components_
        return X_transformed