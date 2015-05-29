import numpy as np

from .wrap import reduction
from . import cudarray
from . import helpers


REDUCE_ALL = 0
REDUCE_LEADING = 1
REDUCE_TRAILING = 2


def reduce_shape(shape, axis, keepdims):
    if keepdims:
        out_shape = list(shape)
        for a in axis:
            out_shape[a] = 1
        return tuple(out_shape)
    all_axis = tuple(range(len(shape)))
    if axis == all_axis:
        return (1,)
    else:
        return tuple(shape[a] for a in all_axis if a not in axis)


def reduce_type(axis, ndim):
    all_axis = tuple(range(ndim))
    if axis == all_axis:
        return REDUCE_ALL
    elif axis == all_axis[:len(axis)]:
        return REDUCE_LEADING
    elif axis == all_axis[-len(axis):]:
        return REDUCE_TRAILING
    raise ValueError('reduction of middle axes not implemented')


def reduce(op, a, axis=None, dtype=None, out=None, keepdims=False,
           to_int_op=False):
    axis = helpers.normalize_axis(axis, a.ndim)
    out_shape = reduce_shape(a.shape, axis, keepdims)

    if to_int_op:
        out_dtype = np.dtype('int32')
    else:
        out_dtype = a.dtype

    if out is None:
        out = cudarray.empty(out_shape, out_dtype)
    else:
        if not out.shape == out_shape:
            raise ValueError('out.shape does not match result')
        if not out.dtype == out_dtype:
            raise ValueError('dtype mismatch')

    rtype = reduce_type(axis, a.ndim)
    if rtype == REDUCE_ALL:
        if to_int_op:
            reduction._reduce_to_int(op, a._data, a.size, out._data)
        else:
            reduction._reduce(op, a._data, a.size, out._data)
    elif rtype == REDUCE_LEADING:
        n = helpers.prod(out_shape)
        m = a.size / n
        if to_int_op:
            reduction._reduce_mat_to_int(op, a._data, m, n, True, out._data)
        else:
            reduction._reduce_mat(op, a._data, m, n, True, out._data)
    else:
        m = helpers.prod(out_shape)
        n = a.size / m
        if to_int_op:
            reduction._reduce_mat_to_int(op, a._data, m, n, False, out._data)
        else:
            reduction._reduce_mat(op, a._data, m, n, False, out._data)
    return out


def amax(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduce(reduction.max_op, a, axis, dtype, out, keepdims)


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduce(reduction.mean_op, a, axis, dtype, out, keepdims)


def amin(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduce(reduction.min_op, a, axis, dtype, out, keepdims)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduce(reduction.sum_op, a, axis, dtype, out, keepdims)


def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduce(reduction.argmax_op, a, axis, dtype, out, keepdims, True)


def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduce(reduction.argmin_op, a, axis, dtype, out, keepdims, True)
