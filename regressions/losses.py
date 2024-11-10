import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    ----------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
    -------
        float: The MSE, calculated as the mean of squared differences between y_true and y_pred.

    Raises:
    ------
        ValueError: If y_true and y_pred have different dimensions.
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("the two inputs y_pred and y_true must have the same dimension")
    
    squared = np.square(y_true - y_pred)
    mean_squared = 0.5*np.mean(a=squared, axis = 0)
    
    return np.sum(a = mean_squared, axis=0)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    ----------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
    -------
        float: The RMSE, calculated as the square root of the MSE.
    """
    return np.sqrt(mse(y_true=y_true, y_pred=y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    ----------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
    -------
        float: The MAE, calculated as the mean of absolute differences between y_true and y_pred.

    Raises:
    ------
        ValueError: If y_true and y_pred have different dimensions.
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        
    if y_true.shape != y_pred.shape:
        raise ValueError("the two inputs y_pred and y_true must have the same dimension")
    diff = np.abs(y_true - y_pred)
    mean_absulue = np.mean(a=diff, axis = 0)
    
    return np.sum(a=mean_absulue, axis = 0)