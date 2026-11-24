import numpy as np
from scipy.optimize import linprog

def lad_fit(X, y):
    N, D = X.shape
    c = np.concatenate((np.zeros(D), np.ones(N)))
    A_ub = np.block([
        [X, - np.eye(N)],
        [- X, - np.eye(N)]
    ])
    b_ub = np.concatenate((y, -y))
    bounds = [(None, None)] * D + [(0, None)] * N
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    v = res.x
    return v[:D]

def huber_fit(X, y, delta, n_iter):
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(n_iter):
        r = y - X @ w

        abs_r = np.abs(r)
        S = np.ones_like(r, dtype=float)
        mask = abs_r > delta
        S[mask] = delta / abs_r[mask]
        
        sqrt_S = np.sqrt(S)
        w, _, _, _ = np.linalg.lstsq(X * sqrt_S[:, None], y * sqrt_S, rcond=None)
    return w
