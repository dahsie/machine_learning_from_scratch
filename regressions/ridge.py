from linear_regression import LinearRegression
import numpy as np


class Ridge(LinearRegression):
    """
    Ridge regression is a type of linear regression that uses L2 regularization 
    to prevent overfitting by penalizing the square of the coefficients. 
    This class extends the LinearRegression class and overrides the gradient descent 
    and cost functions to incorporate the Ridge penalty.

    Attributs:
    ---------
        alpha (float): Regularization strength. Larger values of alpha result in stronger regularization.
        **params: Additional parameters passed to the LinearRegression class.
    """

    def __init__(self, alpha:float = 0.01,**params):
        super().__init__(**params)
        self.alpha = alpha

    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the cost function with L2 regularization (Ridge) during gradient descent.

        This method overrides the gradient descent in LinearRegression by adding the L2 penalty term, 
        which is proportional to the square of the weights.

        Parameters:
        ----------
            X (np.ndarray): The input feature matrix.
            Y_true (np.ndarray): The true target values.
            Y_pred (np.ndarray): The predicted values.

        Returns:
        -------
            np.ndarray: The gradient of the cost function with respect to the weights.
        """
        return super()._gradient_descent(X=X, Y_true=Y_true, Y_pred=Y_pred) + self.alpha * self.W

    def _cost(self,Y_true, Y_pred):
        """
        Computes the cost function with L2 regularization (Ridge) for the given predictions.

        This method overrides the cost function in LinearRegression to include the L2 penalty term, 
        which penalizes the square of the coefficients to prevent overfitting.

        Parameters:
        ----------
            Y_true (np.ndarray): The true target values.
            Y_pred (np.ndarray): The predicted values.

        Returns:
        -------
            float: The cost (error) with L2 regularization.
        """
        return super()._cost(Y_true, Y_pred) +self.alpha * 0.5*np.sum(a=np.square(self.W), axis=0)