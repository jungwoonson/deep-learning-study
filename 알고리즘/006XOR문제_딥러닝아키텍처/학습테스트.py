import numpy as np
import LogicGate as lg

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND 게이트
t_data = np.array([0, 0, 0, 1])
and_obj = lg.LogicGate("AND_GATE", x_data, t_data)
and_obj.train()
print(and_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = and_obj.predict(input_data)
    print(input_data, " = ", logical_val)

# OR 게이트
t_data = np.array([0, 1, 1, 1])
or_obj = lg.LogicGate("OR_GATE", x_data, t_data)
or_obj.train()
print(or_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = or_obj.predict(input_data)
    print(input_data, " = ", logical_val)

# NAND 게이트
t_data = np.array([1, 1, 1, 0])
nand_obj = lg.LogicGate("NAND_GATE", x_data, t_data)
nand_obj.train()
print(nand_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = nand_obj.predict(input_data)
    print(input_data, " = ", logical_val)

# XOR 게이트
t_data = np.array([0, 1, 1, 0])
xor_obj = lg.LogicGate("XOR_GATE", x_data, t_data)
xor_obj.train()
print(xor_obj.name)
for input_data in test_data:
    (sigmoid_val, logical_val) = xor_obj.predict(input_data)
    print(input_data, " = ", logical_val)
