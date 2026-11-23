import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    n, d = X.shape
    ans = np.zeros(n_steps, dtype = float)
    for k in range(n_steps):
        lr_k = 10 ** a if n_steps == 1 else 10 ** (a + (b - a) * k / (n_steps - 1))
        gradient = (2 / n) * X.T @ (X @ w0 - y)
        w0 -= lr_k * gradient
        ans[k] = (1 / n) * (X @ w0 - y).T @ (X @ w0 - y)
    return ans