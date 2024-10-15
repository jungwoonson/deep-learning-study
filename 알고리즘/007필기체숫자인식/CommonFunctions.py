import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def numerical_derivative(f, x):
    delta = 1e-4
    grad = np.zeros_like(x)

    # 다차원 파라미터를 순회하며 각 파라미터에 대해 기울기를 계산
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]

        # f(x+delta) 계산
        x[idx] = original_value + delta
        fx_plus_delta = f()

        # f(x-delta) 계산
        x[idx] = original_value - delta
        fx_minus_delta = f()

        # 기울기 계산
        grad[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta)
        x[idx] = original_value

        it.iternext()

    return grad