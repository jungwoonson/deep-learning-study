import tensorflow as tf
import numpy as np

# 데이터 로드 (CSV 파일에서 데이터를 불러옴, 구분자는 쉼표)
loaded_data = np.loadtxt('./../test_data/data-01.csv', delimiter=',', dtype=np.float32)

# 입력 데이터와 타깃 데이터를 분리 (마지막 열은 타깃 데이터로, 나머지는 입력 데이터로 처리)
x_data = loaded_data[:, 0:-1]  # 입력 데이터 (특성들)
t_data = loaded_data[:, -1].reshape(-1, 1)  # 타깃 데이터 (출력값), 2D 배열로 변환하여 호환성 유지

# 데이터의 차원 출력 (모델에 필요한 입력/출력 데이터의 크기를 확인하기 위함)
print("x_data.shape = ", x_data.shape)
print("t_data.shape = ", t_data.shape)

# 모델 변수 정의 (가중치 W와 편향 b를 임의의 초기값으로 설정)
# W는 3개의 입력 특성에 대해 학습해야 하므로 (3, 1)의 형태를 가짐
W = tf.Variable(tf.random.normal([3, 1]))
# b는 편향(bias)로, 출력값 하나에 대해 하나의 값만 필요함
b = tf.Variable(tf.random.normal([1]))

# 예측 함수 정의 (입력 X에 대해 예측값 y를 계산하는 함수)
# 선형 회귀 모델 y = XW + b
def model(X):
    return tf.matmul(X, W) + b

# 손실 함수 정의 (Mean Squared Error, MSE 사용)
# 실제 타깃값(T)과 모델이 예측한 값(y)의 차이를 제곱하여 평균을 구함
def loss_fn(X, T):
    y = model(X)  # 모델의 예측값
    return tf.reduce_mean(tf.square(y - T))  # MSE 계산

# 옵티마이저 설정 (확률적 경사 하강법(SGD) 사용, 학습률은 1e-5)
learning_rate = 1e-5
optimizer = tf.keras.optimizers.SGD(learning_rate)

# 학습 루프 (8001번 반복하여 모델을 학습)
for step in range(8001):
    # GradientTape를 사용하여 경사 계산을 기록
    with tf.GradientTape() as tape:
        loss_val = loss_fn(x_data, t_data)  # 현재 가중치와 편향으로 손실 값 계산

    # 손실 값에 대한 W와 b의 기울기 계산
    gradients = tape.gradient(loss_val, [W, b])
    # 계산된 기울기를 사용하여 가중치와 편향을 업데이트
    optimizer.apply_gradients(zip(gradients, [W, b]))

    # 학습 과정 모니터링 (400번마다 현재 step과 손실 값 출력)
    if step % 400 == 0:
        print("step = ", step, ", loss_val = ", loss_val.numpy())

# 새로운 입력 데이터에 대해 예측 수행
new_x = np.array([[100, 98, 81]], dtype=np.float32)  # 예시 입력 데이터
prediction = model(new_x)  # 예측값 계산
print("\nPrediction is ", prediction.numpy())  # 예측 결과 출력
