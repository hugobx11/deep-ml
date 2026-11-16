import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for k in range(n):
        a = A[:, k]
        e = a.copy()
        for l in range(k):
            ps = Q[:, l] @ a
            R[l, k] = ps
            e -= ps * Q[:, l]
        Q[:, k] = e / np.linalg.norm(e)
        R[k, k] = Q[:, k] @ a
    return Q, R