import numpy as np
from linear_regression import LinearRegression


class ElasticNet(LinearRegression):

    def __init__(self, alpha:float = 0.01, l1_ratio:float = 0.5,**params):
        super().__init__(**params)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray):
        return super()._gradient_descent(X=X, Y_true=Y_true, Y_pred=Y_pred) + self.l1_ratio*self.alpha*np.sign(self.W) + (1-self.l1_ratio)*self.alpha * self.W

    def _cost(self,Y_true, Y_pred):
        return super()._cost(Y_true, Y_pred) + self.l1_ratio * self.alpha * np.sum(a=np.abs(self.W), axis=0) + 0.5 * self.alpha * (1 - self.l1_ratio)*np.sum(a=np.square(self.W), axis=0)