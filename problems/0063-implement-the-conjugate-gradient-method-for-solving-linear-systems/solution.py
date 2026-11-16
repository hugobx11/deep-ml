import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    iteration = 0
    r = b - A @ x0
    p = r
    for _ in range(n):
        alpha = (r @ r).item() / (p.T @ A @ p).item()
        x0 += alpha * p
        new_r = r - alpha * (A @ p)
        if np.linalg.norm(new_r) < tol:
            break
        beta = (new_r @ new_r).item() / (r @ r).item()
        r = new_r
        p = r + beta * p
    return x0
