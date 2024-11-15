import numpy as np
from discriminant_analysis.discriminant_analysis import DiscriminantAnalysis

class QuadraticDiscriminantAnalysis(DiscriminantAnalysis):
    """
    Predict the class and probabilities for a single sample using LDA.

    Args:
        x (np.ndarray): Input sample.

    Returns:
        tuple: Predicted class index, probabilities, and evidence.
    """
    def __init__(self):
        super().__init__()

    def _predict(self, x):
        """
        Predict the class and probabilities for a single sample using QDA.

        Args:
            x (np.ndarray): Input sample.

        Returns:
            tuple: Predicted class index, probabilities, and evidence.
        """
        probas = np.zeros(len(self.uniques_labels))
        for index in range(len(self.uniques_labels)):
            mean, cov = self._mean[index, :], np.squeeze(self._cov_ij[index,:,:])
            proba = self._mutli_normal_density(x=x,mean=mean, sigma = cov) * self._priors[index]
            probas[index] = proba
        
        evidence = np.sum(probas, axis=0) + 1e-8
        probas = probas / evidence

        return np.argmax(probas), probas, evidence
    