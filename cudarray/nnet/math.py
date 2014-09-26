import numpy as np

from ..elementwise import unary
from ..cudarray_wrap import elementwise as wrap


def relu(x, out=None):
    return unary(wrap.relu_op, x, out)


def relu_d(x, out=None):
    return unary(wrap.relu_d_op, x, out)


def sigmoid(x, out=None):
    return unary(wrap.sigmoid_op, x, out)


def sigmoid_d(x, out=None):
    return unary(wrap.sigmoid_d_op, x, out)


def tanh_d(x, out=None):
    return unary(wrap.tanh_d_op, x, out)
