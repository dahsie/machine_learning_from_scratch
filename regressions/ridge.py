from linear_regression import LinearRegression
import numpy as np


class Ridge(LinearRegression):

    def __init__(self, alpha:float = 0.01,**params):
        super().__init__(**params)
        self.alpha = alpha

    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray):
        return super()._gradient_descent(X=X, Y_true=Y_true, Y_pred=Y_pred) + self.alpha * self.W

    def _cost(self,Y_true, Y_pred):
        return super()._cost(Y_true, Y_pred) +self.alpha * 0.5*np.sum(a=np.square(self.W), axis=0)