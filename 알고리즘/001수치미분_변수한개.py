# 함수 f(x) = x^2를 미분

# f(x) = x^2
def f(x):
    return x ** 2

# 수치 미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)

# x = 3일 때 미분값 계산
result = numerical_derivative(f, 3)

print(result)  # 6.000000000012662
