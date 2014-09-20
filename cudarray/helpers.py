import numpy as np


def normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    elif isinstance(axis, (int, long, float, complex)):
        return (axis,)
    elif isinstance(axis, tuple):
        return sorted(axis)
    else:
        raise ValueError('invalid axis type')
