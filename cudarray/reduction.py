import numpy as np

import cudarray_wrap.reduction as wrap
import base
from .helpers import normalize_axis


REDUCE_ALL = 0
REDUCE_LEADING = 1
REDUCE_TRAILING = 2


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
    all_axis = tuple(range(ndim))
    if axis == all_axis:
        return REDUCE_ALL
    elif axis == all_axis[:len(axis)]:
        return REDUCE_LEADING
    elif axis == tuple(reversed(all_axis))[:len(axis)]:
        return REDUCE_TRAILING
    raise ValueError('reduction of middle axes not implemented')


def reduction(op, a, axis=None, dtype=None, out=None, keepdims=False):
    axis = normalize_axis(axis, a.ndim)
    out_shape = reduce_shape(a.shape, axis, keepdims)
    if out is None:
        out = base.empty(out_shape, a.dtype)
    else:
        if not out_shape == out.shape:
            raise ValueError('out.shape does not match result')
        if not a.dtype == out.dtype:
            raise ValueError('dtype mismatch')
        if a == out:
            raise ValueError('inplace operation not supported')

    rtype = reduce_type(axis, a.ndim)
    if rtype == REDUCE_ALL:
        wrap._reduce(op, a._data, a.size, out._data)
    elif rtype == REDUCE_LEADING:
        n = np.prod(out_shape)
        m = a.size / n
        wrap._reduce_mat(op, a._data, m, n, True, out._data)
    else:
        m = np.prod(out_shape)
        n = a.size / m
        wrap._reduce_mat(op, a._data, m, n, False, out._data)
    return out


def max(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduction(wrap.max_op, a, axis, dtype, out, keepdims)


def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduction(wrap.mean_op, a, axis, dtype, out, keepdims)


def min(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduction(wrap.min_op, a, axis, dtype, out, keepdims)


def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    return reduction(wrap.sum_op, a, axis, dtype, out, keepdims)
