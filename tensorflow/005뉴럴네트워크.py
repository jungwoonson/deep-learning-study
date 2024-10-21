import tensorflow as tf
import numpy as np

# MNIST 데이터셋 불러오기 (훈련 데이터와 테스트 데이터로 분리)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
# 1. 이미지를 1차원 벡터로 변환 (28x28 이미지를 784 픽셀의 벡터로 만듦)
# 2. 각 픽셀 값을 0~1 사이로 정규화 (0~255 범위를 255로 나눔)
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# 레이블을 one-hot 인코딩으로 변환
# 0~9 숫자 레이블을 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 형태로 변환
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 데이터셋 크기 출력 (훈련 데이터 개수와 테스트 데이터 개수)
print("# train examples:", len(x_train), "test examples:", len(x_test))

# 데이터의 shape 출력 (데이터의 차원 확인)
print("train image shape =", np.shape(x_train))  # 훈련 이미지의 형태 출력
print("train label shape =", np.shape(y_train))  # 훈련 레이블의 형태 출력
print("test image shape =", np.shape(x_test))  # 테스트 이미지의 형태 출력
print("test label shape =", np.shape(y_test))  # 테스트 레이블의 형태 출력

# 하이퍼파라미터 설정
learning_rate = 0.1  # 학습률 설정 (가중치가 얼마나 업데이트될지를 결정)
epochs = 100  # 학습 반복 횟수
batch_size = 100  # 한 번의 학습에 사용될 데이터의 개수 (배치 크기)

# 모델 생성
# Sequential API를 사용하여 레이어를 순차적으로 쌓음
model = tf.keras.Sequential([
    # 첫 번째 은닉층 (784 입력 노드 -> 100 은닉 노드)
    # 활성화 함수로 relu 사용 (비선형성을 추가하여 학습 성능 향상)
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),

    # 출력층 (100 은닉 노드 -> 10 출력 노드)
    # softmax 함수로 출력값을 0~9 범위의 확률로 변환
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
# 손실 함수로 categorical_crossentropy 사용 (다중 클래스 분류 문제에 적합)
# SGD(경사하강법)를 사용하여 모델 학습 (learning_rate는 미리 설정된 값 사용)
# 성능 평가를 위해 정확도(accuracy)를 지표로 설정
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
# fit() 함수로 모델을 학습시킴
# x_train: 학습 데이터 (입력)
# y_train: 학습 데이터에 대한 레이블 (정답)
# epochs: 학습 반복 횟수
# batch_size: 배치 크기 (한 번에 처리할 데이터의 수)
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# 모델 평가
# evaluate() 함수로 테스트 데이터에 대한 성능을 평가함
# 테스트 데이터와 레이블을 입력으로 제공하고, 손실값과 정확도를 반환받음
test_loss, test_acc = model.evaluate(x_test, y_test)

# 테스트 정확도 출력
print("\n# Test Accuracy =", test_acc)
