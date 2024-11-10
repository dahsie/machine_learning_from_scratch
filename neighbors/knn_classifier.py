import numpy as np

class KNN:
    """
    Implements a simple K-Nearest Neighbors (KNN) classifier.

    Attributs:
    ----------
        n_neighbors (int): Number of neighbors to consider for classification (default: 3).
        metric (str): Distance metric to use ('euclidean' by default).

    Methods:
    -------
        fit(X, Y): Trains the model by storing the training data and their labels.
        predict(X): Predicts labels for a new set of data points.
    """
    
    def __init__(self, n_neighbors:int =3, metric: str = 'euclidean') -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.train_data = None
        self.train_target = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Stores the training data and corresponding labels.

        Parameters:
        ----------
            X (np.ndarray): Training data, with samples as rows.
            Y (np.ndarray): Labels for the training data.
        """
        self.train_data = X
        self.train_target = Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for each sample in the input data.

        Parameters:
        ----------
            X (np.ndarray): Data to predict labels for, with samples as rows.

        Returns:
        -------
            np.ndarray: Predicted labels for each input sample.
        """
        return np.array([self._predict(x=sample) for sample in X])

    def _predict(self,x: np.ndarray) -> int:
        """
        Predicts the label for a single sample based on the nearest neighbors.

        Parameters:
        ----------
            x (np.ndarray): A single sample to predict the label for.

        Returns:
        -------
            int: Predicted label for the sample.
        """
        distances = None
        if self.metric == 'euclidean':
            distances = np.array([self.euclidean_distance(x1=x, x2=sample) for sample in self.train_data])

        nearest_index = np.argsort(a= distances)[:self.n_neighbors]
        nearest_labels = self.train_target[nearest_index]

        return np.bincount(nearest_labels).argmax()

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculates the Euclidean distance between two samples.

        Parameters:
            x1 (np.ndarray): First sample.
            x2 (np.ndarray): Second sample.

        Returns:
            float: Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2)**2, axis = 0))
    

    # -------------------- Dunder methods -----------------------------------------------------------

    def __str__(self) -> str:
        # Crée un dictionnaire pour représenter les attributs de l'instance sans transformation spéciale
        return f"KNN(n_neighbors={self.n_neighbors}, metric='{self.metric}')"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_neighbors={self.n_neighbors}, metric='{self.metric}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)
