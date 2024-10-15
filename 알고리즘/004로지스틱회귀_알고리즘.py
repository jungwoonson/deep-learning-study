import numpy as np

# 학습 데이터
x_data = np.array([[2, 4], [4, 11], [6, 6], [8, 5], [10, 7], [12, 16], [14, 8], [16, 3], [18, 7]])
t_data = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(9, 1)

# 가중치와 편향을 랜덤하게 초기화
w = np.random.rand(2, 1)
b = np.random.rand(1)

print(f"Initial w = {w}, w.shape = {w.shape}, b = {b}, b.shape = {b.shape}")


# 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 크로스 엔트로피 손실 함수
def cross_entropy_loss(x, t):
    delta = 1e-7  # 로그의 무한대 발산 방지용 작은 값
    z = np.dot(x, w) + b
    y = sigmoid(z)
    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log(1 - y + delta))


# 수치 미분 함수: 파라미터에 대한 기울기 계산
def compute_gradient(f, param):
    delta = 1e-4
    grad = np.zeros_like(param)

    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + delta
        fx_plus_delta = f()

        param[idx] = original_value - delta
        fx_minus_delta = f()

        grad[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta)
        param[idx] = original_value

        it.iternext()

    return grad


# 예측 함수
def predict(x):
    z = np.dot(x, w) + b
    y = sigmoid(z)
    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result  # 배열의 첫 번째 요소를 스칼라로 추출하여 비교


# 학습 함수
def train(steps, learning_rate):
    global w, b  # 함수 내부에서 전역 변수를 수정 가능하게 선언

    loss_function = lambda: cross_entropy_loss(x_data, t_data)

    # 초기 에러 값 출력
    print(f"Initial error = {cross_entropy_loss(x_data, t_data)}, Initial w = {w}, b = {b}")

    for step in range(steps):
        # 가중치와 편향에 대해 기울기 계산 및 업데이트
        w -= learning_rate * compute_gradient(loss_function, w)
        b -= learning_rate * compute_gradient(loss_function, b)

        # 400 스텝마다 중간 결과 출력
        if step % 400 == 0:
            print(f"step = {step}, error value = {cross_entropy_loss(x_data, t_data)}, w = {w}, b = {b}")


# 학습 실행
train(steps=10001, learning_rate=1e-2)

# 테스트 데이터에 대한 예측
(real_val, logical_val) = predict(np.array([17, 3]))
print(f"Predicted probability: {real_val}, Predicted class: {logical_val}")
