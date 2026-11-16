import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    A = np.asarray(A)
    if (
        np.ndim(A) != 2 or
        A.shape[0] != A.shape[1] or 
        not np.allclose(A.T, A)
    ):
        return -1
    l = np.zeros(A.shape)
    n = A.shape[0]
    for i in range(n):
        to_sqrt = A[i, i] - l[i, :i] @ l[i, :i]
        if to_sqrt <= 0:
            return -1
        l[i, i] = np.sqrt(to_sqrt)
        for j in range(i+1, n):
            l[j, i] = (A[i, j] - l[i, :i] @ l[j, :i]) / l[i, i]
    return l
