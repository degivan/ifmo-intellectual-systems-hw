from math import sqrt


class Dot(object):
    def __init__(self, arr):
        self.x = float(arr[0])
        self.y = float(arr[1])
        self.label = arr[2][0]

    def __str__(self):
        return 'x: %f y: %f label: %s' % (self.x, self.y, self.label)


def get_x(dot):
    return dot.x


def get_y(dot):
    return dot.y


def get_square_sum(dot):
    return sqrt((dot.x ** 2 + dot.y ** 2))
