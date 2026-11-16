import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    
    Args:
        A: Square matrix as a list of lists
    
    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    n, m = len(A), len(A[0])
    u = np.zeros((n, n))
    l = np.eye(n)
    for i in range(n):
        u[i, i] = A[i][i] - l[i, :i] @ u[:i, i]
        for j in range(i+1, n):
            u[i, j] = A[i][j] - l[i, :i] @ u[:i, j]
            l[j, i] = (A[j][i] - l[j, :j] @ u[:j, i]) / u[i, i]
    return l, u