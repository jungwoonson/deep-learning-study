import tensorflow as tf

# 상수 노드 정의
a = tf.constant(1.0, name='a')
b = tf.constant(2.0, name='b')
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print('상수 노드 출력')
print(a)
print(a + b)
print(c)

print('상수 노드 연산')
print(a.numpy(), b.numpy())
print(c.numpy())
print((a + b).numpy())
print((c + 1.0).numpy())