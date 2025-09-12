
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def f_asym_wmae(y_true, y_pred, alpha=0.333, wi=1.25):
    """
    Compute Asymmetrical Weighted MAE with perishable weight.
    alpha: penalty for under-forecasting (0.5 = symmetric)
    wi: perishable weight (use 1.25 for all if only perishables)
    """
    errors = y_true - y_pred
    weights = np.where(errors > 0, alpha, 1 - alpha)
    wmae = np.sum(wi * weights * np.abs(errors)) / np.sum(wi * weights)
    return wmae

def f_compute_metrics(y_true, y_pred, alpha=0.333, wi=1.25):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100
    asym_wmae_score = asym_wmae(y_true, y_pred, alpha=alpha, wi=wi)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Asym_WMAE': asym_wmae_score}