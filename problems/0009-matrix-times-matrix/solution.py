def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    n, m = len(a), len(b[0])
    l = len(b)
    c = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(l):
                c[i][j] += a[i][k] * b[k][j]
    return c