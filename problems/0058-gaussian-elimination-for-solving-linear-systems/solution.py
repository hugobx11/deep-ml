import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    n = len(A)
    x = np.zeros(n, dtype=float)
    # Transform A in upper triangular matrix
    for i in range(n):
        # Find the pivot
        r_max = np.argmax(np.absolute(A[i:, i])) + i
        pivot = A[r_max, i]
        if pivot == 0:
            continue
        A[[i, r_max]] = A[[r_max, i]]
        b[[i, r_max]] = b[[r_max, i]]
        # Update rows below the pivot
        for j in range(i+1, n):
            scale = A[j, i] / pivot
            A[j] -= scale * A[i]
            b[j] -= scale * b[i]
    # Backward substitution
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i] 
    return x
