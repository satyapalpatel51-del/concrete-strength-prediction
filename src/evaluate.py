"""
evaluate.py
Shared function to evaluate a trained regression model.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_model(model, X_test, y_test):
    """Return R2, RMSE, MAE for a trained model on test data."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}