from __future__ import print_function

from collections import defaultdict

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
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, r in zip(resY, testY):
        if t == r:
            tp[t] += 1
        else:
            fp[t] += 1
            fn[r] += 1
    for i in range(10):
        precision = tp[i] / float(tp[i] + fp[i])
        recall = tp[i] / float(tp[i] + fn[i])
        f1 = 2 * precision * recall / (precision + recall)
        print("F1-score for {} is {}".format(i, f1))


if __name__ == '__main__':
    trainX, trainY, testX, testY = get_data()

    nn = NeuralNetwork([trainX.shape[1], 112, 10])
    nn.train(trainX, to_vector(trainY), max_iter=500000, learning_rate=0.1)
    resY = nn.predict(testX)

    print_acc(resY, testY)
