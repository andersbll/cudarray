import numpy as np

from ..elementwise import unary
from ..wrap import elementwise


def relu(x, out=None):
    return unary(elementwise.relu_op, x, out)


def relu_d(x, out=None):
    return unary(elementwise.relu_d_op, x, out)


def sigmoid(x, out=None):
    return unary(elementwise.sigmoid_op, x, out)


def sigmoid_d(x, out=None):
    return unary(elementwise.sigmoid_d_op, x, out)


def tanh_d(x, out=None):
    return unary(elementwise.tanh_d_op, x, out)
