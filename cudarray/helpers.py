import numpy as np


def normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    elif isinstance(axis, (int, long)):
        return (axis,)
    elif isinstance(axis, tuple):
        return tuple(sorted(axis))
    else:
        raise ValueError('invalid axis type')


def normalize_shape(shape):
    if isinstance(shape, (int, long)):
        return (shape,)
    elif isinstance(shape, tuple):
        return shape
    else:
        raise ValueError('invalid shape')
