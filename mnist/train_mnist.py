from __future__ import print_function

import numpy as np

from get_mnist import get_data
from neural.perceptron import NeuralNetwork


def to_vector(Y):
    res = []
    for d in Y:
        nxt = np.zeros(10)
        nxt[d] = 1
        res.append(nxt)
    return np.array(res)


def print_acc(resY, testY):
    suc = 0
    for t, r in zip(resY, testY):
        if t == r:
            suc += 1
    print(suc / float(len(testY)))


if __name__ == '__main__':
    trainX, trainY, testX, testY = get_data()

    nn = NeuralNetwork([trainX.shape[1], 28, 10])
    nn.train(trainX, to_vector(trainY), max_iter=400000, learning_rate=0.13)
    resY = nn.predict(testX)

    print_acc(resY, testY)
