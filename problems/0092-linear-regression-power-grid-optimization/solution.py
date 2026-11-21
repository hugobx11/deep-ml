import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
    consumption_data = np.asarray(consumption_data, dtype=float)
    n = consumption_data.shape[0]
    # 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    y = consumption_data - 10 * np.sin(PI * np.arange(1, 11) / 5)
    # 2) Perform linear regression on the detrended data.
    X = np.array([np.ones(n), np.arange(1, n + 1)]).T
    c, slope = np.linalg.inv(X.T @ X) @ X.T @ y
    # 3) Predict day 15's base consumption.
    y_pred = slope * 15 + c
    # 4) Add the day 15 fluctuation back.
    y_pred += 10 * np.sin(PI * 15 / 5)
    # 5) Round, then add a 5% safety margin (rounded up).
    y_pred = np.ceil(y_pred) * 1.05
    # 6) Return the final integer.
    return np.round(y_pred)