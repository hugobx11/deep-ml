import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
    for _ in range(n_epochs):
        if method == 'batch':
            gradient = (2 / m) * X.T @ (X @ weights - y)
            weights -= learning_rate * gradient
        elif method == 'stochastic':
            for i in range(m):
                gradient = 2 * X[i] * (X[i] @ weights - y[i])
                weights -= learning_rate * gradient
        if method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                gradient = (2 / len(X_batch)) * X_batch.T @ (X_batch @ weights - y_batch)
                weights -= learning_rate * gradient
    return weights
