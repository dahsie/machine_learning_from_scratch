import numpy as np

class LogisticRegression:

    def __init__(self, random_state: int = 42, learning_rate: float = 1e-4, max_iterations: int = 1000000, stopping_threshold: float = 1e-6):
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold
        self.W = None
        self.history = []
        

    def _init_params(self,dim: int):
        np.random.seed(self.random_state)
        self.W = np.random.rand(dim +1)

    def sigmoid(self, Z: np.ndarray):
        return 1/(1 + np.exp(-Z))
    
    def predict_proba(self, X: np.ndarray) :
        ones = np.ones((X.shape[0], 1))
        X = np.copy(X)
        X = np.concatenate((ones, X), axis = 1)
        return self.sigmoid(np.dot(X,self.W))
    
    def predict(self,X: np.ndarray):

        Y_pred_proba = self.predict_proba(X)
        mask = Y_pred_proba >= 0.5
        Y_pred = np.zeros(Y_pred_proba.shape[0])
        Y_pred[mask] = 1

        return Y_pred
    
    def _gradient_descent(self, X: np.ndarray, Y_true: np.ndarray, Y_pred_proba: np.ndarray):
        diff = Y_pred_proba - Y_true
        return np.concatenate((np.sum(a=diff, axis = 0, keepdims=True), np.dot(X.T, diff)), axis=0)/X.shape[0]
    

    def _update_params(self,X: np.ndarray,Y_true: np.ndarray,Y_pred_proba: np.ndarray, learning_rate):
        dW= self._gradient_descent(X=X,Y_true=Y_true, Y_pred_proba=Y_pred_proba)
        self.W = self.W - learning_rate * dW
    
    def _cost(self,Y_true, Y_pred_proba):
        return -np.sum(a =Y_true * np.log(Y_pred_proba) + (1 - Y_true)*np.log(Y_pred_proba), axis=0)/Y_true.shape[0]
    
    def fit(self, X: np.ndarray, Y: np.ndarray):

        n, p = X.shape

        self._init_params(p)
        previous_cost = None
        
        for itr in range(self.max_iterations):

            Y_pred_proba = self.predict_proba(X)
            
            cost = self._cost(Y_true=Y, Y_pred_proba=Y_pred_proba)
            self.history.append(cost)

            # if previous_cost is not None and abs(previous_cost-cost)<=self.stopping_threshold:
                # break
            previous_cost = cost
            self._update_params(X=X, Y_true=Y, Y_pred_proba=Y_pred_proba, learning_rate=self.learning_rate)
    
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