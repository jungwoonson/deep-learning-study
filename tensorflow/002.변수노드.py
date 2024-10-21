import tensorflow as tf

# 값이 계속 업데이트될 변수노드 정의
W1 = tf.Variable(tf.random.normal([1]))  # W1 = np.random.rand(1) 비슷함
b1 = tf.Variable(tf.random.normal([1]))  # b1 = np.random.rand(1) 비슷함

W2 = tf.Variable(tf.random.normal([1, 2]))  # W2 = np.random.rand(1, 2)
b2 = tf.Variable(tf.random.normal([1, 2]))  # b2 = np.random.rand(1, 2)

# TensorFlow 2.0에서는 Session과 global_variables_initializer()가 필요 없음
for step in range(3):
    # W1, b1, W2, b2 값을 직접 업데이트
    W1.assign(W1 - step)  # W1 변수 노드 업데이트
    b1.assign(b1 - step)  # b1 변수 노드 업데이트
    W2.assign(W2 - step)  # W2 변수 노드 업데이트
    b2.assign(b2 - step)  # b2 변수 노드 업데이트

    print("Step =", step, ", W1 =", W1.numpy(), ", b1 =", b1.numpy())
    print("Step =", step, ", W2 =", W2.numpy(), ", b2 =", b2.numpy())
