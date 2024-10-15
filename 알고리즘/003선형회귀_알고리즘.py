import numpy as np

# 데이터셋 준비: 입력 데이터(x_data)와 목표 데이터(t_data)
x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
t_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

# 파라미터 초기화: 가중치(weight)와 편향(bias)
weight = np.random.rand(1, 1)
bias = np.random.rand(1)

# 초기 파라미터 값 출력
print(f"Initial weight = {weight}, weight.shape = {weight.shape}, bias = {bias}, bias.shape = {bias.shape}")

# 예측 함수: 입력 데이터에 대해 예측값을 반환
def predict(x):
    return np.dot(x, weight) + bias

# 손실 함수: 평균 제곱 오차(MSE) 계산
def compute_loss(x, t):
    prediction = predict(x)
    return np.mean((t - prediction) ** 2)

# 수치 미분 함수: 파라미터에 대한 기울기 계산
def numerical_derivative(f, param):
    delta = 1e-4
    grad = np.zeros_like(param)

    # 다차원 파라미터를 순회하며 각 파라미터에 대해 기울기를 계산
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        # f(x+delta) 계산
        param[idx] = original_value + delta
        fx_plus_delta = f()

        # f(x-delta) 계산
        param[idx] = original_value - delta
        fx_minus_delta = f()

        # 기울기 계산
        grad[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta)
        param[idx] = original_value

        it.iternext()

    return grad

# 초기 에러 값 출력 함수
def print_initial_error():
    error = compute_loss(x_data, t_data)
    print(f"Initial error = {error}, Initial weight = {weight}, bias = {bias}")

# 학습 함수: 가중치와 편향을 업데이트
def train(x, t, learning_rate, steps):
    loss_function = lambda: compute_loss(x, t)

    for step in range(steps):
        # 가중치와 편향에 대한 기울기 계산 및 업데이트
        global weight, bias
        weight -= learning_rate * numerical_derivative(loss_function, weight)
        bias -= learning_rate * numerical_derivative(loss_function, bias)

        # 400번마다 현재 에러 값과 파라미터 출력
        if step % 400 == 0:
            error = compute_loss(x, t)
            print(f"Step = {step}, Error = {error}, weight = {weight}, bias = {bias}")

# 학습률 설정 및 학습 실행
learning_rate = 1e-2
steps = 8001

# 초기 에러 값 출력
print_initial_error()

# 학습 시작
train(x_data, t_data, learning_rate, steps)

# 학습 후 예측값 출력
predicted_value = predict(43)
print(f"Predicted value for 43 = {predicted_value}")
