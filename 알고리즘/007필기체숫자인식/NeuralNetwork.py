import numpy as np

import CommonFunctions as cf


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_data = None
        self.target_data = None

        self.input_nodes = input_nodes  # 784
        self.hidden_nodes = hidden_nodes  # 100
        self.output_nodes = output_nodes  # 10

        # 2층 hidden layer unit
        # 가중치 w, 바이어스 b 초기화
        self.w2 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.b2 = np.random.rand(self.hidden_nodes)

        # 3층 output layer unit
        self.w3 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.b3 = np.random.rand(self.output_nodes)

        self.learning_rate = 1e-4

    # feed forward 를 이용하여 입력층에서부터 출력층까지의 데이터를 전달하고 손실 함수 값 계산
    # loss_val(self) 메서드와 동일한 코드 - 외부 출력용으로 사용
    def __feed_forward(self):
        delta = 1e-7

        z1 = np.dot(self.input_data, self.w2) + self.b2
        y1 = cf.sigmoid(z1)

        z2 = np.dot(y1, self.w3) + self.b3
        y = cf.sigmoid(z2)

        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    def loss_val(self):
        return self.__feed_forward()

    def train(self, training_data):
        # normalize
        # one-hot encoding 을 위한 10개의 노드 0.01 초기화 및 정답을 나타내는 인덱스에 가장 큰 값인 0.99 설정
        self.target_data = np.zeros(self.output_nodes) + 0.01
        self.target_data[int(training_data[0])] = 0.99

        # 입력 데이터는 0~255 이기 때문에, 가끔 overflow 발생. 따라서 모든 입력 값을 0~1사이로 normalize
        self.input_data = (training_data[1:] / 255.0 * 0.99) + 0.01

        f = lambda: self.__feed_forward()

        self.w2 -= self.learning_rate * cf.numerical_derivative(f, self.w2)
        self.b2 -= self.learning_rate * cf.numerical_derivative(f, self.b2)
        self.w3 -= self.learning_rate * cf.numerical_derivative(f, self.w3)
        self.b3 -= self.learning_rate * cf.numerical_derivative(f, self.b3)

    def predict(self, input_data):
        z1 = np.dot(input_data, self.w2) + self.b2
        y1 = cf.sigmoid(z1)

        z2 = np.dot(y1, self.w3) + self.b3
        y = cf.sigmoid(z2)

        predicted_num = np.argmax(y)

        return predicted_num

    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(test_data)):
            label = int(test_data[index, 0])  # Test Data 의 정답 분리

            # normalize
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
            predicated_num = self.predict(data)

            if label == predicated_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)

        print(f"Current Accuracy = ", 100 * (len(matched_list) / len(test_data)))

        return matched_list, not_matched_list
