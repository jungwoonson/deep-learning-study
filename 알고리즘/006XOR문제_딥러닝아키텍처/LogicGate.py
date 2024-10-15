import numpy as np


class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4, 2)  # 4개의 입력 데이터 x1, x2에 대하여 batch 처리 행렬
        self.__tdata = tdata.reshape(4, 1)  # 4개의 입력 데이터 x1, x2에 대한 각각의 계산 값

        # 2층 hidden layer unit: 6개, 가중치 w2, 바이어스 b2 초기화
        self.__w2 = np.random.rand(2, 6)
        self.__b2 = np.random.rand(6)

        # 3층 output layer unit: 1개, 가중치 w3, 바이어스 b3 초기화
        self.__w3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        # 학습률 learning rate 초기화
        self.__learning_rate = 1e-2

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __feed_forward(self):
        delta = 1e-7
        z2 = np.dot(self.__xdata, self.__w2) + self.__b2  # 은닉층의 선형회귀 값
        a2 = self.__sigmoid(z2)  # 은틱층의 출력

        z3 = np.dot(a2, self.__w3) + self.__b3
        y = self.__sigmoid(z3)

        ## cross-entropy
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) + delta))

    def loss_val(self):
        return self.__feed_forward()

    def numerical_derivative(self, f, x):
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

    def train(self):
        f = lambda: self.__feed_forward()
        print("Initial loss value = ", self.loss_val())
        for step in range(10001):
            self.__w2 -= self.__learning_rate * self.numerical_derivative(f, self.__w2)
            self.__b2 -= self.__learning_rate * self.numerical_derivative(f, self.__b2)
            self.__w3 -= self.__learning_rate * self.numerical_derivative(f, self.__w3)
            self.__b3 -= self.__learning_rate * self.numerical_derivative(f, self.__b3)

            if step % 400 == 0:
                print("step = ", step, "error value = ", self.loss_val())

    def predict(self, test_data):
        z2 = np.dot(test_data, self.__w2) + self.__b2
        a2 = self.__sigmoid(z2)
        z3 = np.dot(a2, self.__w3) + self.__b3
        a3 = self.__sigmoid(z3)
        y = a3
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result
