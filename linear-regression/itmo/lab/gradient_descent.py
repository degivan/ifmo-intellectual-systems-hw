from numpy import *


def step_gradient(b1, b2, m, param, learning_rate):
    pass


def run(learning_rate, num_iterations, points):
    [b1, b2, m] = [0, 0, 0]
    for i in range(num_iterations):
        [b1, b2, m] = step_gradient(b1, b2, m, array(points), learning_rate)
    return [b1, b2, m]

if __name__ == '__main__':
    points = genfromtxt("../../prices.txt", dtype=(int, int, int), delimiter=",")

    [b1, b2, m, error] = run(0.001, 1000, points)

    print(b1, b2, m, error)
