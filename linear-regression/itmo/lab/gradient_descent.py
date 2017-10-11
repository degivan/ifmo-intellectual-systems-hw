# coding=utf-8
from numpy import *

# ищем решение вида b1 * area + b2 * rooms + m = price

def compute_error(b1, b2, m, points):
    totalError = 0
    for i in range(0, len(points)):
        area = points[i][0]
        rooms = points[i][1]
        price = points[i][2]
        totalError += (price - (b1 * area + b2 * rooms + m)) ** 2
    return sqrt(totalError / float(len(points)))


def step_gradient(b1, b2, m, points, learning_rate):
    b1_grad = 0
    b2_grad = 0
    m_grad = 0
    coeff = 2 / float(len(points))
    for i in range(0, len(points)):
        area = points[i][0]
        rooms = points[i][1]
        price = points[i][2]
        b1_grad += coeff * area * (b1 * area + rooms * b2 + m - price)
        b2_grad += coeff * rooms * (b1 * area + rooms * b2 + m - price)
        m_grad += coeff * (b1 * area + rooms * b2 + m - price)
    new_b1 = b1 - (learning_rate * b1_grad)
    new_b2 = b2 - (learning_rate * b2_grad)
    new_m = m - (learning_rate * m_grad)
    return [new_b1, new_b2, new_m]


def run(learning_rate, num_iterations, points):
    [b1, b2, m] = [60, 30000, 100000]
    for i in range(num_iterations):
        [b1, b2, m] = step_gradient(b1, b2, m, array(points), learning_rate)
    return [b1, b2, m]

if __name__ == '__main__':
    points = genfromtxt("../../prices.txt", dtype=[float64, float64, float64], delimiter=",")
    print(compute_error(60, 30000, 100000, array(points)))
    [b1, b2, m] = run(1e-6, 10 ** 5, points)
    print(b1, b2, m)
    print(compute_error(b1, b2, m, array(points)))