import numpy as np

def min_norm_solution(A, b):
    y = np.linalg.solve(A @ A.T, b)
    return A.T @ y

def min_norm_via_kkt(A, b):
    m, n = A.shape
    M = np.block([
        [2 * np.eye(n), A.T],
        [A, np.zeros((m, m))]
    ])
    y = np.block([np.zeros(n), b])
    result = np.linalg.solve(M, y)
    return result[:n], result[n:]

def min_norm_from_particular(A, x_p):
    b = A @ x_p
    return min_norm_solution(A, b)
