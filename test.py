import numpy as np
import time

def time_it(func):
    def inner_func(*args):
        start = time.perf_counter()
        result = func(args[0], args[1])
        print(time.perf_counter() - start)
        return result
    return inner_func

@time_it
def A(dims: int, dx):
    A = np.identity(dims)
    for i in range(dims):
        for j in range(dims):
            if i == j:
                A[i, j] = 2
            elif i == j +1 or i == j - 1:
                A[i, j] = -1
    A[0, 0] = 1
    A[dims-1, dims -1] = 1
    return A * 1/dx

@time_it
def A_np(dims: int, dx):
    A = np.identity(dims) * 2 + np.diag(-np.ones(dims - 1), k = 1) + np.diag(-np.ones(dims - 1), k = -1)
    A[0, 0] = 1
    A[dims - 1, dims - 1] = 1

    return A * 1/dx


a = np.random.random((10, 10))
b = np.ones((10, 10)) * 0.4
print(np.linalg.norm(a-b)/100)

print(np.median(a))
# print(A(2000, 1))
# print(A_np(2000, 1))