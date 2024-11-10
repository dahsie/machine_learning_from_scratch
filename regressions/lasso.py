from linear_regression import LinearRegression
import numpy as np

class Lasso(LinearRegression):
    """
    Lasso regression is a type of linear regression that uses L1 regularization 
    to prevent overfitting by penalizing the absolute size of the coefficients.
    This class extends the LinearRegression class and overrides the gradient descent 
    and cost functions to incorporate the Lasso penalty.

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
        Computes the gradient of the cost function with L1 regularization (Lasso) during gradient descent.

        This method overrides the gradient descent in LinearRegression by adding the L1 penalty term, 
        which is proportional to the absolute value of the weights.

        Parameters:
        ----------
            X (np.ndarray): The input feature matrix.
            Y_true (np.ndarray): The true target values.
            Y_pred (np.ndarray): The predicted values.

        Returns:
        -------
            np.ndarray: The gradient of the cost function with respect to the weights.
        """
        return super()._gradient_descent(X=X, Y_true=Y_true, Y_pred=Y_pred) + self.alpha * np.sign(self.W)

    def _cost(self,Y_true, Y_pred) -> float:
        """
        Computes the cost function with L1 regularization (Lasso) for the given predictions.

        This method overrides the cost function in LinearRegression to include the L1 penalty term, 
        which penalizes the absolute values of the coefficients to prevent overfitting.

        Parameters:
        ---------
            Y_true (np.ndarray): The true target values.
            Y_pred (np.ndarray): The predicted values.

        Returns:
        -------
            float: The cost (error) with L1 regularization.
        """
        return super()._cost(Y_true, Y_pred) +self.alpha * np.sum(a=np.abs(self.W), axis=0)
    