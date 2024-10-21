import tensorflow as tf
import numpy as np

# 'gohome' Data Creation
idx2char = ['g', 'o', 'h', 'm', 'e']   # 각 문자를 인덱스로 변환 (g = 0, o = 1, h = 2, m = 3, e = 4)

x_data = [[0, 1, 2, 1, 3]]   # "gohom"에 해당하는 인덱스 리스트

x_one_hot = [[[1, 0, 0, 0, 0],    # g = 0
              [0, 1, 0, 0, 0],    # o = 1
              [0, 0, 1, 0, 0],    # h = 2
              [0, 1, 0, 0, 0],    # o = 1
              [0, 0, 0, 1, 0]]]   # m = 3, 입력을 one-hot 벡터로 변환한 것

t_data = [[1, 2, 1, 3, 4]]   # "ohome"에 해당하는 정답 인덱스 리스트

num_classes = 5   # 예측할 문자의 총 클래스 수 (g, o, h, m, e)
input_dim = 5     # one-hot 벡터의 크기 (5개의 클래스)
hidden_size = 5   # RNN 레이어의 출력 크기, 즉 hidden units의 개수
batch_size = 1    # 배치 크기 (한 문장씩 입력)
sequence_length = 5   # 문장 길이 ("gohom" -> 5글자)
learning_rate = 0.1   # 학습률

# NumPy 배열을 사용하여 입력 데이터를 정의 (TensorFlow 2.x에서는 placeholder를 사용하지 않음)
X = np.array(x_one_hot, dtype=np.float32)
T = np.array(t_data, dtype=np.int32)

# RNN 레이어 정의: SimpleRNN은 기본적인 RNN 구조이며, hidden_size만큼의 유닛을 가짐.
# return_sequences=True는 각 타임스텝의 출력을 반환하도록 설정함.
# activation='tanh'은 하이퍼볼릭 탄젠트 활성화 함수를 사용함.
rnn = tf.keras.layers.SimpleRNN(units=hidden_size, return_sequences=True, activation='tanh')

# Dense 레이어: RNN의 출력을 받아서 num_classes(5개의 클래스)로 변환
fc = tf.keras.layers.Dense(units=num_classes)

# 모델 클래스 정의
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = rnn  # RNN 레이어
        self.fc = fc    # Fully connected(Dense) 레이어

    def call(self, x):
        x = self.rnn(x)   # RNN 레이어를 통해 순차적 데이터를 처리
        x = self.fc(x)    # Dense 레이어를 통해 각 타임스텝에 대해 클래스 예측
        return x

# 모델 인스턴스 생성
model = MyModel()

# 손실 함수: 다중 클래스 분류를 위한 손실 함수, SparseCategoricalCrossentropy 사용.
# from_logits=True로 설정하여 예측값이 소프트맥스 적용 전의 로짓(logits) 값임을 명시.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 최적화 알고리즘: Adam Optimizer 사용, 학습률은 0.1로 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 한 번의 학습 스텝을 수행하는 함수 정의
def train_step(x, t):
    with tf.GradientTape() as tape:
        predictions = model(x)    # 모델을 통해 예측값 계산
        loss = loss_object(t, predictions)  # 손실 함수 계산 (예측값과 실제 라벨 비교)
    gradients = tape.gradient(loss, model.trainable_variables)  # 손실에 따른 그래디언트 계산
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 가중치 업데이트
    return loss, predictions

# 학습 루프
for step in range(2001):
    loss, predictions = train_step(X, T)  # 학습 스텝 수행

    if step % 400 == 0:
        print(f"step = {step}, loss = {loss.numpy()}")  # 현재 스텝과 손실 값 출력

        # 예측값에서 가장 높은 확률을 가진 클래스 인덱스 반환 (axis=2는 각 타임스텝에 대해 수행)
        result = tf.argmax(predictions, axis=2).numpy()

        # 예측된 인덱스를 실제 문자로 변환
        result_str = [idx2char[c] for c in np.squeeze(result)]  # squeeze로 불필요한 차원 제거 후 변환
        print("#Prediction =", ''.join(result_str))  # 예측된 문자 시퀀스를 출력
