import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    N, D = X.shape
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    m = len(batch_indices)
    gradient_w = (2 / m) * X_batch.T @ (X_batch @ weights + bias - y_batch)
    derivate_b = (2 / m) * sum((X_batch @ weights + bias - y_batch))
    weights -= lr * gradient_w
    bias -= lr * derivate_b
    return np.concatenate([weights, [bias]])
