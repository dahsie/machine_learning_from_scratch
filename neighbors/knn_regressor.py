import numpy as np
from knn import KNN

class KNNRegressor(KNN):
    """
    Implements a simple K-Nearest Neighbors (KNN) regessor.

    Attributs:
    ----------
        n_neighbors (int): Number of neighbors to consider for regressor (default: 3).
        metric (str): Distance metric to use ('euclidean' by default).

    Methods:
    -------
        fit(X, Y): Trains the model by storing the training data and their labels.
        predict(X): Predicts values for a new set of data points.
    """
    
    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _predict(self,x: np.ndarray) -> float:
        """
        Predicts the label for a single sample based on the nearest neighbors.

        Parameters:
        ----------
            x (np.ndarray): A single sample to predict the label for.

        Returns:
        -------
            int: Predicted flot for the sample.
        """
        distances = None
        if self.metric == 'euclidean':
            distances = np.array([self.euclidean_distance(x1=x, x2=sample) for sample in self.train_data])

        nearest_index = np.argsort(a= distances)[:self.n_neighbors]
        predicted_value = np.mean(a=self.train_target[nearest_index], axis=0)

        return predicted_value