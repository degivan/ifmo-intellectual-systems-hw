# coding=utf-8
from math import sqrt

from numpy import *


# ищем решение вида b1 * area + b2 * rooms + m = price

def compute_error(b1, b2, m, points, max_price):
    total_error = 0
    for i in range(0, len(points)):
        area = points[i][0]
        rooms = points[i][1]
        price = points[i][2]
        total_error += (price - (b1 * area + b2 * rooms + m)) ** 2
    return max_price * sqrt(total_error / float(len(points)))


def grad_part(point, coeff, b1, b2, m):
    area = point[0]
    rooms = point[1]
    price = point[2]
    coeff_diff = coeff * (b1 * area + rooms * b2 + m - price)
    return [coeff_diff * area, coeff_diff * rooms, coeff_diff]


def step_gradient(b1, b2, m, points, learning_rate):
    coeff = 2 / float(len(points))
    grads = array([grad_part(point, coeff, b1, b2, m) for point in points])
    [b1_grad, b2_grad, m_grad] = grads.sum(axis=0)
    new_b1 = b1 - (learning_rate * b1_grad)
    new_b2 = b2 - (learning_rate * b2_grad)
    new_m = m - (learning_rate * m_grad)
    return [new_b1, new_b2, new_m]


def run(init_b1, init_b2, init_m, learning_rate, num_iterations, points):
    [b1, b2, m] = [init_b1, init_b2, init_m]
    for i in range(num_iterations):
        [b1, b2, m] = step_gradient(b1, b2, m, array(points), learning_rate)
    return [b1, b2, m]


if __name__ == '__main__':
    points = genfromtxt("../../prices.txt", dtype=[float64, float64, float64], delimiter=",")

    max_area = max([x[0] for x in points])
    max_rooms = max([x[1] for x in points])
    max_price = max([x[2] for x in points])
    points = [(x[0] / max_area, x[1] / max_rooms, x[2] / max_price) for x in points]

    [b1, b2, m] = run(1, 0, 1, 1e-4, 10 ** 6, points)
    print([max_price * x for x in [b1, b2, m]])
    print(compute_error(b1, b2, m, array(points), max_price))
