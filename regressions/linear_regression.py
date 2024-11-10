
import numpy as np
from losses import mse

class LinearRegression:

    def __init__(self, random_state: int = 42, learning_rate: float = 1e-4, max_iterations: int = 1000000, stopping_threshold: float = 1e-6):
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold
        self.W = None
        self.coef, self.intercept = None, None
        self.history = []
        

    def _init_params(self,dim: int):
        np.random.seed(self.random_state)
        self.W = np.random.rand(dim +1)

    def predict(self, X: np.ndarray) :
        ones = np.ones((X.shape[0], 1))
        X = np.copy(X)
        X = np.concatenate((ones, X), axis = 1)
        return np.dot(X,self.W)
    
    def score(self, X: np.ndarray, Y):
        y_pred = self.predict(X)
        u = ((Y - y_pred)**2).sum(axis=0)
        v= ((Y - Y.mean(axis=0))**2).sum(axis=0)
        return 1 - (u/v)
    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred: np.ndarray):
        diff = Y_pred - Y_true
        # return np.concatenate((np.sum(a=diff, axis = 0, keepdims=True), np.sum(diff[:, np.newaxis] * X, axis= 0)))/m
        return np.concatenate((np.sum(a=diff, axis = 0, keepdims=True), np.dot(X.T, diff)), axis=0)/X.shape[0]
    

    def _update_params(self,X: np.ndarray,Y_true: np.ndarray,Y_pred: np.ndarray, learning_rate):
        
        dW= self._gradient_descent(X=X,Y_true=Y_true, Y_pred=Y_pred)
        self.W = self.W - learning_rate * dW
    
    def _cost(self,Y_true, Y_pred):
        return mse(y_true=Y_true, y_pred=Y_pred)
    
    def fit(self, X: np.ndarray, Y: np.ndarray):

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

    