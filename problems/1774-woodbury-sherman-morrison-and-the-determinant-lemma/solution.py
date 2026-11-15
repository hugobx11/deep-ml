import numpy as np
from numpy.linalg import multi_dot

def woodbury_solve(sigma_diag, X, B):
    inv_s = 1.0 / sigma_diag
    inv_s_X = inv_s[:, None] * X

    inv_C = np.linalg.inv(
        np.eye(X.shape[1]) + X.T @ inv_s_X
    )

    inv_s_B = (B.T * inv_s).T

    second_term = inv_s_X @ (inv_C @ (inv_s_X.T @ B))
    
    return inv_s_B - second_term

def sherman_morrison(A_inv, u, v):
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    denom = 1 + multi_dot([v.T, A_inv, u]).item()
    if denom == 0:
        raise ValueError("Erreur")
    numerator = multi_dot([A_inv, u, v.T, A_inv])
    return A_inv - numerator / denom

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    scalaire = 1 + multi_dot([v.T, A_inv, u]).item()
    if scalaire <= 0:
        raise ValueError("Erreur")
    return logdet_A + np.log(scalaire)
