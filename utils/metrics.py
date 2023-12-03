
# Import modules
import numpy as np


def RSE(pred, true):
    """Root squared error.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: RSE.
    """
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    """Correlation.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: correlation.
    """
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    """Mean absolute error.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: MAE.
    """
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    """Mean squared error.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: MSE.
    """
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    """Root mean squared error.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: RMSE.
    """
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """Mean absolute percentage error.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: MAPE.
    """
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    """Mean squared percentage error.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: MSPE.
    """
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    """Metrics computation.

    Args:
        pred (list): predictions.
        true (list): ground truth.

    Returns:
        float: MAE, MSE, RMSQ, MAPE, MSPE.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
