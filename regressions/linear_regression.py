
import numpy as np
from losses import mse

class LinearRegression:
    """
    Implements a simple Linear Regression model using gradient descent.

    Attributs:
    ----------
        random_state (int): Seed for random number generation, ensuring reproducibility (default: 42).
        learning_rate (float): Step size for gradient descent updates (default: 1e-4).
        max_iterations (int): Maximum number of iterations for gradient descent (default: 1,000,000).
        stopping_threshold (float): Minimum cost difference between iterations to stop training (default: 1e-6).

    Methods:
    -------
        fit(X, Y): Trains the model on the input data and labels.
        predict(X): Predicts target values for new data based on the trained model.
        score(X, Y): Evaluates the model's accuracy using the R-squared metric.
    """
     
    def __init__(self, random_state: int = 42, learning_rate: float = 1e-4, max_iterations: int = 1000000, stopping_threshold: float = 1e-6) -> None:
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold
        self.W = None
        self.coef, self.intercept = None, None
        self.history = []
        

    def _init_params(self,dim: int) -> None:
        """
        Initializes model parameters randomly.

        Parameters:
        ----------
            dim (int): Dimension of the input features.
        """
        np.random.seed(self.random_state)
        self.W = np.random.rand(dim +1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given input data.

        Parameters:
        ----------
            X (np.ndarray): Input data, with samples as rows.

        Returns:
        -------
            np.ndarray: Predicted target values for each input sample.
        """
        ones = np.ones((X.shape[0], 1))
        X = np.copy(X)
        X = np.concatenate((ones, X), axis = 1)
        return np.dot(X,self.W)
    
    def score(self, X: np.ndarray, Y):
        """
        Calculates the R-squared score of the model.

        Parameters:
        ----------
            X (np.ndarray): Input data.
            Y (np.ndarray): True target values.

        Returns:
        -------
            float: R-squared score indicating the proportion of variance explained by the model.
        """
        y_pred = self.predict(X)
        u = ((Y - y_pred)**2).sum(axis=0)
        v= ((Y - Y.mean(axis=0))**2).sum(axis=0)
        return 1 - (u/v)
    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray):
        """
        Calculates the gradient of the cost function with respect to the model parameters.

        Parameters:
        ----------
            X (np.ndarray): Input data.
            Y_true (np.ndarray): True target values.
            Y_pred (np.ndarray): Predicted target values.

        Returns:
        -------
            np.ndarray: Gradient values for updating model parameters.
        """
        diff = Y_pred - Y_true
        # return np.concatenate((np.sum(a=diff, axis = 0, keepdims=True), np.sum(diff[:, np.newaxis] * X, axis= 0)))/m
        return np.concatenate((np.sum(a=diff, axis = 0, keepdims=True), np.dot(X.T, diff)), axis=0)/X.shape[0]
    

    def _update_params(self,X: np.ndarray,Y_true: np.ndarray,Y_pred: np.ndarray, learning_rate):
        """
        Updates model parameters using gradient descent.

        Parameters:
        ----------
            X (np.ndarray): Input data.
            Y_true (np.ndarray): True target values.
            Y_pred (np.ndarray): Predicted target values.
            learning_rate (float): Step size for updating parameters.
        """
        dW= self._gradient_descent(X=X,Y_true=Y_true, Y_pred=Y_pred)
        self.W = self.W - learning_rate * dW
    
    def _cost(self,Y_true, Y_pred):
        """
        Calculates the mean squared error (MSE) cost.

        Parameters:
        ----------
            Y_true (np.ndarray): True target values.
            Y_pred (np.ndarray): Predicted target values.

        Returns:
        -------
            float: Mean squared error between Y_true and Y_pred.
        """
        return mse(y_true=Y_true, y_pred=Y_pred)
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Trains the model on the input data using gradient descent.

        Parameters:
        ----------
            X (np.ndarray): Training data, with samples as rows.
            Y (np.ndarray): Target values for each training sample.
        """
        n, p = X.shape

        self._init_params(p)
        previous_cost = None
        
        for itr in range(self.max_iterations):

            Y_pred = self.predict(X)
            
            cost = self._cost(Y_true=Y, Y_pred=Y_pred)
            self.history.append(cost)

            if previous_cost is not None and abs(previous_cost-cost)<=self.stopping_threshold:
                break
            previous_cost = cost
            self._update_params(X=X, Y_true=Y, Y_pred=Y_pred, learning_rate=self.learning_rate)
    
    # -------------------- Properties ---------------------------------------------------------------
    @property
    def coef(self):
        return self.W[1:]
    
    @property
    def intercept(self):
        return self.W[0]
    
    # -------------------- Dunter methods -----------------------------------------------------------
    def __str__(self):
        # Crée un dictionnaire en excluant l'attribut 'history'
        attributes = {k: v for k, v in self.__dict__.items() if k != 'history'}
        return str(attributes)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False
    
    def __call__(self, X: np.ndarray):
        return self.predict(X)

    