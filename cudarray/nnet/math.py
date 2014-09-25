import numpy as np

from ..elementwise import unary


def relu(x, out=None):
    return unary(wrap.UnaryOp.relu, x, out)


def relu_d(x, out=None):
    return unary(wrap.UnaryOp.relu_d, x, out)


def sigmoid(x, out=None):
    return unary(wrap.UnaryOp.sigmoid, x, out)


def sigmoid_d(x, out=None):
    return unary(wrap.UnaryOp.sigmoid_d, x, out)


def tanh_d(x, out=None):
    return unary(wrap.UnaryOp.tanh_d, x, out)
