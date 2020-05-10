import numpy as np
from math import e


def relu(X, derivative=False):
    if derivative:
        t = []
        for lst in X:
            t.append(np.array([0 if i <= 0 else 1 for i in lst]))
        return np.array(t)
    return np.array([np.maximum(0, i) for i in X])


def leakyReLU(X, a=0.01, derivative=False):
    if derivative:
        raise NotImplementedError
    return [i if i >= 0 else i * a for i in X]


def sigmoid(X, derivative=False):
    X = np.array(X)
    if derivative:
        return np.array([1 / (1 + e ** (-i)) * (1 - (1 / (1 + e ** (-i)))) for i in X])
    return np.array([1/(1+np.exp(-i)) for i in X])


def softmax(X, derivative=False):
    if derivative:
        raise NotImplementedError
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum


# Example use cases
# print(sigmoid([np.array([1, 5]), np.array([0.9, -30, 8])], True))
# print(sigmoid([1, 5], True))
# print(sigmoid([0.9, -30, 8], True))
# print(relu([[6, 3], [0, -9], [13, -4, -2]], True))
# print(relu([6, 7, -9 ,0], True))

