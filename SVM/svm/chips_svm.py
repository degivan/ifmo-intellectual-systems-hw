from random import shuffle

import numpy as np
from dots import *


def get_features(dot, rules):
    return [rule(dot) for rule in rules]


def get_X(dots):
    return np.array([get_features(d, [get_x, get_y, get_square_sum]) for d in dots])


def get_Y(dots):
    return np.array([d.label for d in dots])


def cross_validate(size, folds=5):
    results = []
    indexes = range(size)
    shuffle(indexes)
    fold_size = size / folds
    for i in range(folds):
        left_index = i * fold_size
        right_index = (i + 1) * fold_size
        test_data = indexes[left_index:right_index]
        train_data = [x for x in indexes if x not in test_data]
        results.append((train_data, test_data))
    return results


def filter_index(X, index):
    return [X[i] for i in index]


if __name__ == '__main__':
    f = file('../chips.txt')
    dots = [Dot(line.split(',')) for line in f.readlines()]
    X = get_X(dots)
    Y = get_Y(dots)
    for train_index, test_index in cross_validate(len(Y)):
        train_X = filter_index(X, train_index)
        train_Y = filter_index(Y, train_index)
        test_X = filter_index(X, test_index)
        test_Y = filter_index(Y, test_index)
        # only classification and metrics left