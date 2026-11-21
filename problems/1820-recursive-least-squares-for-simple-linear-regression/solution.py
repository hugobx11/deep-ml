import numpy as np

def online_simple_regression(xs, ys):
    # return (N, 2) array of (w0, w1) after each point
    ans = np.empty((len(xs), 2), dtype=float)

    mean_x = 0.0
    mean_y = 0.0
    m_xx = 0.0
    m_xy = 0.0
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        dx = x - mean_x
        dy = y - mean_y

        mean_x += dx / i
        mean_y += dy / i

        m_xx += dx * (x - mean_x)
        m_xy += dx * (y - mean_y)

        w1 = 0.0 if m_xx == 0.0 else m_xy / m_xx
        w0 = mean_y - w1 * mean_x
        ans[i - 1] = (w0, w1)
    return ans
