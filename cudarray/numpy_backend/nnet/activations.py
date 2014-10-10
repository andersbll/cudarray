import numpy as np


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_d(x):
    s = sigmoid(x)
    return s*(1-s)


def tanh_d(x):
    return 1-np.tanh(x)**2


def relu(x):
    return np.maximum(0.0, x)


def relu_d(x):
    dx = np.zeros(x.shape)
    dx[x >= 0] = 1
    return dx
