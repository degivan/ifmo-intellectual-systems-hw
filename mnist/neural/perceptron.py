import random

from neural.activation import *


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.activation = sigmoid
        self.dactivation = dsigmoid
        self.w = [np.random.randn(self.layers[i + 1], self.layers[i]) for i in range(self.num_layers - 1)]

    def predict(self, X):
        res = []
        for x in X:
            self._forward(x)
            res.append(np.argmax(self._u[-1]))
        return res

    def train(self, X, y, max_iter=100000, learning_rate=0.5):
        for j in range(max_iter):
            curid = random.randint(0, len(X) - 1)
            self._forward(X[curid])
            self._backward(X[curid], y[curid], learning_rate)
            if j % 10000 == 0:
                print("Iteration: {}".format(str(j)))

    def _forward(self, x):
        self._u = [x]
        for i in range(self.num_layers - 1):
            self._u.append(self.activation(self.w[i].dot(self._u[i])))

    def _backward(self, x, y, learning_rate):
        eps = [self._u[-1] - y]
        for i in reversed(range(1, self.num_layers - 1)):
            tmp = eps[-1] * self.dactivation(self._u[i + 1])
            eps.append(self.w[i].T.dot(tmp))
        eps = list(reversed(eps))
        for i in range(self.num_layers - 1):
            self.w[i] = self.w[i] - learning_rate * np.outer(eps[i] * self.dactivation(self._u[i + 1]), self._u[i])
