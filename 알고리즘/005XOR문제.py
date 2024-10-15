import numpy as np

class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        # 가중치 w, 바이어스 b 초기화
        self.__w = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        # 학습률 learning rate 초기화
        self.__learning_rate = 1e-2

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 손실함수
    def __loss_func(self):
        delta = 1e-7  # 무한대 발산 방지

        z = np.dot(self.__xdata, self.__w) + self.__b
        y = self.__sigmoid(z)

        return - np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log((1 - y) + delta))

    def error_val(self):
        return self.__loss_func()

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
        f = lambda: self.__loss_func()
        print("Initial error value = ", self.error_val())
        for step in range(8001):
            self.__w -= self.__learning_rate * self.numerical_derivative(f, self.__w)
            self.__b -= self.__learning_rate * self.numerical_derivative(f, self.__b)

            if step % 400 == 0:
                print("step = ", step, "error value = ", self.error_val())

    def predict(self, test_data):
        z = np.dot(test_data, self.__w) + self.__b
        y = self.__sigmoid(z)
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND 게이트
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_data = np.array([0, 0, 0, 1])
AND_obj = LogicGate("AND_GATE", x_data, t_data)
AND_obj.train()
print(AND_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, " = ", logical_val)

# OR 게이트
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_data = np.array([0, 1, 1, 1])
OR_obj = LogicGate("OR_GATE", x_data, t_data)
OR_obj.train()
print(OR_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print(input_data, " = ", logical_val)

# NAND 게이트
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_data = np.array([1, 1, 1, 0])
NAND_obj = LogicGate("NAND_GATE", x_data, t_data)
NAND_obj.train()
print(NAND_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print(input_data, " = ", logical_val)

# XOR 게이트: AND, OR, NAND 게이트를 조합하여 구현
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

s1 = [] # NAND 출력
s2 = [] # OR 출력

new_input_data = [] # AND 입력
final_output = [] # AND 출력

for index in range(len(input_data)):
    s1 = NAND_obj.predict(input_data[index])
    s2 = OR_obj.predict(input_data[index])

    new_input_data.append(s1[-1])
    new_input_data.append(s2[-1])

    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))

    final_output.append(logical_val)
    new_input_data = []

print("XOR_GATE")
for index in range(len(input_data)):
    print(input_data[index], " = ", final_output[index])