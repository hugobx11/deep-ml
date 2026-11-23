import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    n = x.shape[0]
    numerical_grad = np.zeros_like(x, dtype=float)
    for i in range(analytical_grad.shape[0]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    norm_diff = np.linalg.norm(numerical_grad - analytical_grad)
    norm_sum = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    if norm_sum == 0:
        relative_error = 0.0
    else:
        relative_error = float(norm_diff / norm_sum)

    return numerical_grad, relative_error