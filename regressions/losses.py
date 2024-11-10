import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray):

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("the two inputs y_pred and y_true must have the same dimension")
    
    squared = np.square(y_true - y_pred)
    mean_squared = 0.5*np.mean(a=squared, axis = 0)
    
    return np.sum(a = mean_squared, axis=0)

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(mse(y_true=y_true, y_pred=y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray):

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        
    if y_true.shape != y_pred.shape:
        raise ValueError("the two inputs y_pred and y_true must have the same dimension")
    diff = np.abs(y_true - y_pred)
    mean_absulue = np.mean(a=diff, axis = 0)
    
    return np.sum(a=mean_absulue, axis = 0)