import tensorflow as tf
import numpy as np

# 데이터 로드 (CSV 파일에서 데이터 읽기, 콤마로 구분)
loaded_data = np.loadtxt('./diabetes.csv', delimiter=',')

# x_data: 입력 데이터 (마지막 열을 제외한 모든 열), t_data: 출력 라벨 (마지막 열)
x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("loaded_data = ", loaded_data.shape)
print("x_data = ", x_data.shape, ", t_data = ", t_data.shape)

# 사용자 정의 모델 클래스 정의 (tf.keras.Model을 상속받음)
class DiabetesModel(tf.keras.Model):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        # 밀집(Dense) 레이어를 정의, 뉴런 1개, 시그모이드 활성화 함수 사용
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    # 모델이 호출될 때 실행되는 함수 (입력 데이터를 받아서 예측 값을 출력)
    def call(self, x):
        # 입력값 x를 밀집 레이어에 통과시켜 예측 값을 반환
        return self.dense(x)


# 데이터셋을 텐서로 변환 (tf.placeholder 대신 사용됨)
X = tf.convert_to_tensor(x_data, dtype=tf.float32)
T = tf.convert_to_tensor(t_data, dtype=tf.float32)

# 모델 인스턴스 생성
model = DiabetesModel()

# 손실 함수: 이진 교차 엔트로피(Binary Crossentropy)를 사용 (이진 분류 문제에서 사용)
loss_object = tf.keras.losses.BinaryCrossentropy()

# 옵티마이저: 확률적 경사 하강법(SGD) 사용, 학습률(learning rate)은 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 학습 손실 값과 정확도를 추적하기 위한 메트릭 정의
train_loss = tf.keras.metrics.Mean(name='train_loss')  # 손실 값의 평균을 계산
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')  # 정확도를 이진 분류 방식으로 계산

# 학습 단계 정의, 각 학습 배치에서 실행되는 과정
@tf.function  # 이 함수는 그래프 모드로 컴파일되어 성능이 최적화됨
def train_step(inputs, targets):
    # GradientTape로 역전파를 위한 계산 기록
    with tf.GradientTape() as tape:
        # 모델에 입력 데이터를 전달하여 예측 값 계산
        predictions = model(inputs)
        # 실제 값과 예측 값 사이의 손실 계산 (이진 교차 엔트로피)
        loss = loss_object(targets, predictions)

    # 손실에 대한 가중치의 기울기(gradient) 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    # 계산된 기울기를 사용하여 모델의 가중치 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 학습 중 손실 값 및 정확도 메트릭 업데이트
    train_loss(loss)
    train_accuracy(targets, predictions)

# 학습 실행
EPOCHS = 20001  # 20001번 학습 반복
for step in range(EPOCHS):
    # 각 학습 단계마다 손실과 정확도를 업데이트
    train_step(X, T)

    # 500번마다 손실과 정확도 출력
    if step % 500 == 0:
        print(f"Step {step}: Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}")

# 최종 예측 값 및 정확도 확인
predicted_val = model(X)
accuracy_val = train_accuracy(T, predicted_val)

print("Predicted values = ", predicted_val.numpy().shape)
print("Accuracy = ", accuracy_val.numpy())
