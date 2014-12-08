import numpy as np


def _output(result, out):
    if out is None:
        return result
    else:
        np.copyto(out, result)
        return out


def sigmoid(x, out=None):
    result = 1.0/(1.0+np.exp(-x))
    return _output(result, out)


def sigmoid_d(x, out=None):
    s = sigmoid(x)
    result = s*(1-s)
    return _output(result, out)

def tanh(x, out=None):
    result = np.tanh(x)
    return _output(result, out)

def tanh_d(x, out=None):
    result = 1-np.tanh(x)**2
    return _output(result, out)


def relu(x, out=None):
    result = np.maximum(0.0, x)
    return _output(result, out)


def relu_d(x, out=None):
    result = np.zeros(x.shape)
    result[x >= 0] = 1
    return _output(result, out)
