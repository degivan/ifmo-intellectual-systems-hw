import numpy as np


def sigmoid(M):
    return 1 / (1 + np.exp(-M))


def dsigmoid(M):
    return sigmoid(M) - sigmoid(M) ** 2


def leaky_relu(M, a=0.01):
    if M.all() < 0:
        return a * M
    else:
        return M


def dleaky_relu(M, a=0.01):
    if M.all() < 0:
        return a
    else:
        return 1


def softplus(M):
    return np.log(1 + np.exp(M))


def dsoftplus(M):
    return 1 / (1 + np.exp(-M))


def tanh(M):
    return np.tanh(M)


def dtanh(M):
    return 1 - tanh(M) ** 2


def arctan(M):
    return np.arctan(M)


def darctan(M):
    return 1 / (M ** 2 + 1)
