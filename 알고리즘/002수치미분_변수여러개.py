import numpy as np

# 입력 변수 2개인 함수 f(x, y) = 2x + 3xy + y^3
def f(input_obj):
    x = input_obj[0]
    y = input_obj[1]
    return 2 * x + 3 * x * y + np.power(y, 3)

# 수치 미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4 # 작은 변화값
    grad = np.zeros_like(x) # x와 같은 크기의 배열을 0으로 초기화

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    # 모든 매개변수에 대해 기울기 계산
    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

# x = 1, y = 2일 때 미분값 계산
result = numerical_derivative(f, np.array([1.0, 2.0]))

print(result)