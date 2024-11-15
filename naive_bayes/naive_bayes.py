
import numpy as np


class GaussianNaiveBayes:

    def __init__(self):
        
        self.__mean: np.ndarray = np.empty((0, 0))
        self.__std: np.ndarray = np.empty((0, 0))
        self.__priors: np.ndarray = np.array([])
        self.__is_fitted = False
        self.__propas = np.empty((0, 0))
        self.n_sample, self.n_features_in = 0, 0
        self.uniques_labels = np.array([])



    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:

        labels =np.unique(Y)
        self.uniques_labels = labels
        n, p = X.shape
        self.n_sample, self.n_features_in = n, p

        self.__mean = np.empty((0,p))
        self.__std = np.empty((0,p))
        self.__priors = np.zeros(len(labels))

        if not self.__is_fitted:
            for index,label in enumerate(labels):
                data = X[Y==label, :]

                self.__priors[index] = len(data)/n
                self.__mean = np.concatenate((self.__mean,np.mean(a= data, axis=0, keepdims=True)), axis=0)

                std = np.std(a=data, axis=0, keepdims=True)
                std[std==0] = np.inf
                self.__std = np.concatenate((self.__std, std), axis=0)

        self.__is_fitted = True

    def gaussian_density(self,x, mean, std) -> np.ndarray:
        const = std * np.sqrt(2*np.pi)
        return np.exp(-0.5*((x - mean)/std)**2) / const
    
    def _predict(self, x):

        probas = np.zeros(len(self.uniques_labels))
        for index in range(len(self.uniques_labels)):
            mean, std =self.__mean[index, :], self.__std[index, :]
            proba = self.gaussian_density(x=x,mean=mean, std=std).prod() * self.__priors[index]
            probas[index] = proba
        
        evidence = np.sum(probas, axis=0)
        probas = probas / evidence

        return np.argmax(probas), probas, evidence
    
    def predict(self, X: np.ndarray):

        if not self.__is_fitted:
            raise ValueError("The model must be first trained before using it to make prediction")
        
        n, p = X.shape
        labels = np.zeros(n)
        self.__propas = np.empty((0,len(self.uniques_labels)))
        for index in range(n):
            label, proba, _ = self._predict(X[index,:])
            labels[index] = label
            self.__propas = np.concatenate((proba[np.newaxis,:],self.__propas), axis=0)
        return labels


    # ------------- Methods dunder ------------------------------------

    def __repr__(self):
        """Technical representation for debugging."""
        return (
            f"NaiveBayes(n_samples={self.n_sample}, n_features={self.n_features_in}, "
            f"labels={self.uniques_labels})"
        )

    def __str__(self):
        """Readable string representation for the user."""
        return (
            f"NaiveBayes Classifier:\n"
            f"- Number of samples: {self.n_sample}\n"
            f"- Number of features: {self.n_features_in}\n"
            f"- Unique labels: {self.uniques_labels}\n"
            f"- Fitted: {self.__is_fitted}"
        )
    
    def __call__(self, X:np.ndarray):
        return self.predict(X=X)

        
    #------------- Properties ------------------------------------

    @property
    def mean(self):
        return self.__mean
    @property
    def std(self):
        return self.__std
    @property
    def priors(self):
        return self.__priors
    @property
    def probas(self):
        return self.__propas
