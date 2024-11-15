
import numpy as np

from discriminant_analysis.discriminant_analysis import DiscriminantAnalysis

class LinearDiscriminantAnalysis(DiscriminantAnalysis):

    """
    Linear Discriminant Analysis (LDA) model.
    """

    def __init__(self):
        super().__init__()

    def _predict(self, x):

        probas = np.zeros(len(self.uniques_labels))
        for index in range(len(self.uniques_labels)):
            mean = self._mean[index, :]
            proba = self._mutli_normal_density(x=x,mean=mean, sigma = self._cov) * self._priors[index]
            probas[index] = proba
        
        evidence = np.sum(probas, axis=0) + 1e-8
        probas = probas / evidence

        return np.argmax(probas), probas, evidence
    