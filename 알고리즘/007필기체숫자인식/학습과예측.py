import numpy as np
import NeuralNetwork as nn

# http://www.pjreddie.com/media/files/mnist_train.csv
training_data = np.loadtxt('../../test_data/mnist_train.csv', delimiter=',', dtype=np.float32)
# http://www.pjreddie.com/media/files/mnist_test.csv
test_data = np.loadtxt('../../test_data/mnist_test.csv', delimiter=',', dtype=np.float32)

print(f"training_data.shape = {training_data.shape}, test_data.shape = {test_data.shape}")

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

nn_obj = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

for step in range(0, 30001):
    # 총 60,000 개의 테이터 중 임의의 30,000개 데이터 선별
    index = np.random.randint(0, len(training_data) - 1)

    nn_obj.train(training_data[index])

    print(f"step = {step}, loss_val = {nn_obj.loss_val()}")

nn_obj.accuracy(test_data)