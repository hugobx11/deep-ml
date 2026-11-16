import numpy as np

def forward_sub(L, b):
    n = L.shape[1]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def back_sub(U, y):
    n = U.shape[1]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i:] @ x[i:]) / U[i, i]
    return x

def chol_solve(A, b):
    L = np.linalg.cholesky(A)
    y = forward_sub(L, b)
    return back_sub(L.T, y)

def logdet_chol(A):
    L = np.linalg.cholesky(A)
    return 2 * sum(np.log(L[i, i]) for i in range(L.shape[0]))

def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)
    ans = np.zeros(eps.shape)
    for i in range(ans.shape[0]):
        ans[i] = L @ eps[i] + mu
    return ans
