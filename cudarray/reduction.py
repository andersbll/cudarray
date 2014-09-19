import numpy as np

import cudarray_wrap.reduction as wrap
import cudarray


REDUCE_ALL = 0
REDUCE_LEADING = 1
REDUCE_TRAILING = 2


def normalize_axis(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    elif isinstance(axis, (int, long, float, complex)):
        return (axis,)
    elif isinstance(axis, tuple):
        return sorted(axis)
    else:
        raise ValueError('invalid axis type')


def reduce_shape(shape, axis, keepdims):
    all_axis = tuple(range(len(shape)))
    if axis == all_axis:
        return (1,)
    if keepdims:
        out_shape = shape
        for a in axis:
            out_shape[a] = 1
        return out_shape
    else:
        return tuple(shape[a] for a in all_axis if a not in axis)


def reduce_type(axis, ndim):
    if axis == tuple(range(ndim)):
        return REDUCE_ALL
    elif axis[0] == 0:
        return REDUCE_LEADING
    elif axis[-1] == ndim-1:
        return REDUCE_TRAILING
    raise ValueError('reduction of middle axes not implemented')


def sum(a, axis=None, dtype=None, out=None, keepdims=False):

    axis = normalize_axis(axis, a.ndim)
    out_shape = reduce_shape(a.shape, axis, keepdims)
    if out is None:
        out = cudarray.empty(out_shape, a.dtype)
    else:
        if not out_shape == out.shape:
            raise ValueError('out.shape does not match result')
        if not a.dtype == out.dtype:
            raise ValueError('dtype mismatch')
        if a == out:
            raise ValueError('inplace operation not supported')

    rtype = reduce_type(axis, a.ndim)
    if rtype == REDUCE_ALL:
        wrap.sum(a, out)
    elif rtype == REDUCE_LEADING:
        n = np.prod(out_shape)
        m = a.size / n
        wrap._sum_batched(a, m, n, True, out)
    else:
        m = np.prod(out_shape)
        n = a.size / m
        wrap._sum_batched(a, m, n, False, out)
    return out
