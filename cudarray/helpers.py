from functools import reduce
import operator


def normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    elif isinstance(axis, int):
        return (axis,)
    elif isinstance(axis, tuple):
        return tuple(sorted(axis))
    else:
        raise ValueError('invalid axis type')


def normalize_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    elif isinstance(shape, tuple):
        return shape
    else:
        raise ValueError('invalid shape')


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def require_iterable(x):
    if hasattr(x, '__iter__'):
        return x
    else:
        return [x]
