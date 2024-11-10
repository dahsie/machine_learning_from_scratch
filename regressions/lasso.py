from linear_regression import LinearRegression
import numpy as np

class Lasso(LinearRegression):

    def __init__(self, alpha:float = 0.01,**params):
        super().__init__(**params)
        self.alpha = alpha

    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray):
        return super()._gradient_descent(X=X, Y_true=Y_true, Y_pred=Y_pred) + self.alpha * np.sign(self.W)

    def _cost(self,Y_true, Y_pred):
        return super()._cost(Y_true, Y_pred) +self.alpha * np.sum(a=np.abs(self.W), axis=0)
    