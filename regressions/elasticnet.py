import numpy as np
from linear_regression import LinearRegression


class ElasticNet(LinearRegression):
    """
    Elastic Net regression is a linear regression model that combines L1 and L2 regularization. 
    It is particularly useful when there are multiple features that are correlated. This class 
    extends the LinearRegression class and overrides the gradient descent and cost functions to 
    incorporate the Elastic Net penalty.

    Parameters:
        alpha (float): The regularization strength. Higher values mean stronger regularization.
        l1_ratio (float): The mix ratio between L1 and L2 penalties. A value of 1.0 corresponds to 
                          Lasso regression, while 0.0 corresponds to Ridge regression.
        **params: Additional parameters passed to the LinearRegression class.
    """
    def __init__(self, alpha:float = 0.01, l1_ratio:float = 0.5,**params):
        super().__init__(**params)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the cost function with Elastic Net regularization during gradient descent.

        This method overrides the gradient descent in LinearRegression by adding the combined L1 and L2 
        penalty terms, where `l1_ratio` controls the contribution of each penalty.

        Parameters:
        ----------
            X (np.ndarray): The input feature matrix.
            Y_true (np.ndarray): The true target values.
            Y_pred (np.ndarray): The predicted values.

        Returns:
        -------
            np.ndarray: The gradient of the cost function with respect to the weights.
        """
        return super()._gradient_descent(X=X, Y_true=Y_true, Y_pred=Y_pred) + self.l1_ratio*self.alpha*np.sign(self.W) + (1-self.l1_ratio)*self.alpha * self.W

    def _cost(self,Y_true, Y_pred) -> float:
        """
        Computes the cost function with Elastic Net regularization for the given predictions.

        This method overrides the cost function in LinearRegression to include both L1 and L2 penalties 
        based on the `l1_ratio` parameter.

        Parameters:
        ----------
            Y_true (np.ndarray): The true target values.
            Y_pred (np.ndarray): The predicted values.

        Returns:
        -------
            float: The cost (error) with Elastic Net regularization.
        """
        return super()._cost(Y_true, Y_pred) + self.l1_ratio * self.alpha * np.sum(a=np.abs(self.W), axis=0) + 0.5 * self.alpha * (1 - self.l1_ratio)*np.sum(a=np.square(self.W), axis=0)