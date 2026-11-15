import numpy as np
from numpy.linalg import multi_dot

def partitioned_inverse(M, k):
    n = len(M)
    E = M[0:k, 0:k]
    F = M[0:k, k:]
    G = M[k:, 0:k]
    H = M[k:, k:]
    invH = np.linalg.inv(H)
    SchurH = E - multi_dot([F, invH, G])
    invSchurH = np.linalg.inv(SchurH)
    return np.block([
        [invSchurH, - multi_dot([invSchurH, F, invH])],
        [- multi_dot([invH, G, invSchurH]), invH + multi_dot([invH, G, invSchurH, F, invH])]
    ])


def block_slogdet(M, k):
    E = M[0:k, 0:k]
    F = M[0:k, k:]
    G = M[k:, 0:k]
    H = M[k:, k:]
    invH = np.linalg.inv(H)
    SchurH = E - multi_dot([F, invH, G])
    sign_S, logdet_S = np.linalg.slogdet(SchurH)
    sign_H, logdet_H = np.linalg.slogdet(H)
    return sign_S * sign_H, logdet_S + logdet_H

def bordered_inverse(A_inv, b, c):
    b = np.asarray(b).reshape(-1, 1)
    Schur = c - multi_dot([b.T, A_inv, b]).item()
    if np.abs(Schur) < 1e-12:
        raise ValueError("Erreur")
    invSchur = 1.0 / Schur
    
    top_left = A_inv + invSchur * multi_dot([A_inv, b, b.T, A_inv])
    top_right = -invSchur * np.dot(A_inv, b)
    bottom_left = -invSchur * np.dot(b.T, A_inv)
    bottom_right = np.array([[invSchur]])
    return np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])
