import numpy as np
from numpy import linalg
from abc import ABC, abstractmethod


class DiscriminantAnalysis(ABC):
    """
    Base class for discriminant analysis models.

    Attributes:
        _mean (np.ndarray): Mean vectors of each class.
        _std (np.ndarray): Standard deviation of each feature (not used currently).
        _cov_ij (np.ndarray): Class-specific covariance matrices.
        _cov (np.ndarray): Pooled covariance matrix.
        _priors (np.ndarray): Prior probabilities for each class.
        _is_fitted (bool): Indicates if the model is trained.
        _propas (np.ndarray): Class probabilities for the samples.
        n_sample (int): Number of samples in the dataset.
        n_features_in (int): Number of features in the dataset.
        uniques_labels (np.ndarray): Unique class labels.
    """

    def __init__(self):
        self._mean: np.ndarray = np.empty((0, 0))
        self._std: np.ndarray = np.empty((0, 0))
        self._cov_ij: np.ndarray = np.empty((0,0, 0))
        self._cov: np.ndarray = np.empty((0, 0))
        self._priors: np.ndarray = np.array([])
        self._is_fitted = False
        self._propas = np.empty((0, 0))
        self.n_sample, self.n_features_in = 0, 0
        self.uniques_labels = np.array([])


    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the discriminant analysis model.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            Y (np.ndarray): Target vector of shape (n_samples,).
        """
        labels =np.unique(Y)
        self.uniques_labels = labels
        n, p = X.shape
        self.n_sample, self.n_features_in = n, p

        self._mean = np.empty((0,p))
        self._cov = np.zeros((p, p))
        self._cov_ij = np.empty((0,p, p))
        self._priors = np.zeros(len(labels))

        if not self._is_fitted:
            for index,label in enumerate(labels):
                data = X[Y==label, :]

                self._priors[index] = len(data)/n
                self._mean = np.concatenate((self._mean,np.mean(a= data, axis=0, keepdims=True)), axis=0)

                cov = linalg.pinv(np.cov(data.T))
                self._cov_ij = np.concatenate((self._cov_ij, cov[np.newaxis, :,:]), axis=0)
                self._cov += cov

        self._is_fitted = True

    def _mutli_normal_density(self,x, mean, sigma) -> np.ndarray:
        """
        Compute the density of the multivariate normal distribution.

        Args:
            x (np.ndarray): Input vector.
            mean (np.ndarray): Mean vector of the distribution.
            sigma (np.ndarray): Covariance matrix of the distribution.

        Returns:
            np.ndarray: Density value for the input vector.
        """

        const =  np.sqrt(linalg.det(sigma)) * (2*np.pi)**(x.shape[0]/2)
        distance = (x-mean).T.dot((sigma.dot(x-mean)))
        return np.exp(-0.5* distance) / const
    
    @abstractmethod
    def _predict(self, x):
        """
        Abstract method to predict the class of a single sample.

        Args:
            x (np.ndarray): Input sample.

        Returns:
            Abstract method to be implemented by subclasses.
        """
        pass
    
    def predict(self, X: np.ndarray):
        """
        Predict the class labels for a given dataset.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        if not self._is_fitted:
            raise ValueError("The model must be first trained before using it to make prediction")
        
        n, p = X.shape
        labels = np.zeros(n)
        self._propas = np.empty((0,len(self.uniques_labels)))
        for index in range(n):
            label, proba, _ = self._predict(X[index,:])
            labels[index] = label
            self._propas = np.concatenate((proba[np.newaxis,:],self._propas), axis=0)
        return labels
    

    # ---------------- Dunder methods --------------------------

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_samples={self.n_sample}, "
            f"n_features={self.n_features_in}, labels={self.uniques_labels.tolist()}, "
            f"fitted={self._is_fitted})"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} Classifier:\n"
            f"- Number of samples: {self.n_sample}\n"
            f"- Number of features: {self.n_features_in}\n"
            f"- Unique labels: {self.uniques_labels.tolist()}\n"
            f"- Fitted: {self._is_fitted}"
        )
    
    def __call__(self, X:np.ndarray):
        return self.predict(X=X)
         
    #------------- Properties ------------------------------------

    @property
    def mean(self):
        return self._mean
    @property
    def cov(self):
        return self._cov
    @property
    def cov_ij(self):
        return self._cov_ij
    @property
    def priors(self):
        return self._priors
    @property
    def probas(self):
        return self._propas
    
