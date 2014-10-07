import numpy as np


import cudarray_wrap.elementwise as wrap
import base


NO_BROADCAST = 0
BROADCAST_TO_LEADING = 1
BROADCAST_TO_TRAILING = 2


def broadcast_shape(shape1, shape2):
    if len(shape1) > len(shape2) or np.prod(shape1) > np.prod(shape2):
        return shape1
    else:
        return shape2


def broadcast_type(shape1, shape2):
    if shape1 == shape2:
        return NO_BROADCAST

    error = ValueError('operands could not be broadcast together with shapes '
                       + str(shape1) + ' ' + str(shape2))

    # Bring shapes to same length by setting missing trailing dimensions to 1's
    len_diff = len(shape1) - len(shape2)
    if len_diff > 0:
        shape2 = (1,)*len_diff + shape2
    elif len_diff < 0:
        shape1 = (1,)*(-len_diff) + shape1

    # Find out which axes to broadcast
    b_axes = []
    for a_idx, (a1, a2) in enumerate(zip(shape1, shape2)):
        if a1 != a2:
            if a1 == 1 or a2 == 1:
                b_axes.append(a_idx)
            else:
                raise error

    ndim = len(shape1)
    if b_axes == range(len(b_axes)):
        return BROADCAST_TO_LEADING
    elif b_axes == range(ndim-len(b_axes), ndim):
        return BROADCAST_TO_TRAILING
    else:
        raise error


def binary(op, x1, x2, out=None):
    if np.isscalar(x1) or np.isscalar(x2):
        if np.isscalar(x1):
            scalar = x1
            array = x2
        else:
            array = x1
            scalar = x2

        # Create/check output array
        out_shape = array.shape
        if out is None:
            out = base.empty(out_shape, dtype=x1.dtype)
        else:
            if out_shape != out.shape:
                raise ValueError('out.shape does not match result')
            if array.dtype != out.dtype:
                raise ValueError('dtype mismatch')
        n = array.size
        wrap._binary_scalar(op, array._data, scalar, n, out._data)
        return out

    # Create/check output array
    out_shape = broadcast_shape(x1.shape, x2.shape)
    if x1.dtype == x2.dtype == np.dtype('int32'):
        out_dtype = np.dtype('int32')
    else:
        out_dtype = np.dtype('float32')
    if out is None:
        out = base.empty(out_shape, dtype=out_dtype)
    else:
        if out_shape != out.shape:
            raise ValueError('out.shape does not match result')
        if out_dtype != out.dtype:
            raise ValueError('dtype mismatch')

    btype = broadcast_type(x1.shape, x2.shape)
    if btype == NO_BROADCAST:
        n = x1.size
        wrap._binary(op, x1._data, x2._data, n, out._data)
        return out

    # Calculate dimensions of the broadcast operation
    size1 = x1.size
    size2 = x2.size
    if size1 > size2:
        m, n = size1/size2, size2
    else:
        n, m = size1, size2/size1
        x1, x2 = x2, x1
    b_to_l = btype == BROADCAST_TO_LEADING
    wrap._binary_broadcast(op, x1._data, x2._data, m, n, b_to_l, out._data)
    return out


def add(x1, x2, out=None):
    return binary(wrap.add_op, x1, x2, out)


def subtract(x1, x2, out=None):
    return binary(wrap.sub_op, x1, x2, out)


def multiply(x1, x2, out=None):
    return binary(wrap.mul_op, x1, x2, out)


def divide(x1, x2, out=None):
    return binary(wrap.div_op, x1, x2, out)


def power(x1, x2, out=None):
    return binary(wrap.pow_op, x1, x2, out)


def maximum(x1, x2, out=None):
    return binary(wrap.max_op, x1, x2, out)


def minimum(x1, x2, out=None):
    return binary(wrap.min_op, x1, x2, out)


def binary_cmp(op, x1, x2, out=None):
    out_dtype = np.dtype('int32')
    if np.isscalar(x1) or np.isscalar(x2):
        if np.isscalar(x1):
            scalar = x1
            array = x2
        else:
            array = x1
            scalar = x2

        # Create/check output array
        out_shape = array.shape
        if out is None:
            out = base.empty(out_shape, dtype=out_dtype)
        else:
            if out_shape != out.shape:
                raise ValueError('out.shape does not match result')
            if array.dtype != out.dtype:
                raise ValueError('dtype mismatch')
        n = array.size
        wrap._binary_cmp_scalar(op, array._data, scalar, n, out._data)
        return out

    # Create/check output array
    out_shape = broadcast_shape(x1.shape, x2.shape)
    if out is None:
        out = base.empty(out_shape, dtype=out_dtype)
    else:
        if out_shape != out.shape:
            raise ValueError('out.shape does not match result')
        if out_dtype != out.dtype:
            raise ValueError('dtype mismatch')

    btype = broadcast_type(x1.shape, x2.shape)
    if btype == NO_BROADCAST:
        n = x1.size
        wrap._binary_cmp(op, x1._data, x2._data, n, out._data)
        return out

    # Calculate dimensions of the broadcast operation
    size1 = x1.size
    size2 = x2.size
    if size1 > size2:
        m, n = size1/size2, size2
    else:
        n, m = size1, size2/size1
        x1, x2 = x2, x1
    b_to_l = btype == BROADCAST_TO_LEADING
    wrap._binary_cmp_broadcast(op, x1._data, x2._data, m, n, b_to_l, out._data)
    return out


def equal(x1, x2, out=None):
    return binary_cmp(wrap.eq_op, x1, x2, out)


def greater(x1, x2, out=None):
    return binary_cmp(wrap.gt_op, x1, x2, out)


def greater_equal(x1, x2, out=None):
    return binary_cmp(wrap.gt_eq_op, x1, x2, out)


def less(x1, x2, out=None):
    return binary_cmp(wrap.lt_op, x1, x2, out)


def less_equal(x1, x2, out=None):
    return binary_cmp(wrap.lt_eq_op, x1, x2, out)


def not_equal(x1, x2, out=None):
    return binary_cmp(wrap.neq_op, x1, x2, out)


def unary(op, x, out=None):
    out_shape = x.shape
    if out is None:
        out = base.empty(out_shape, dtype=x.dtype)
    else:
        if not out_shape == out.shape:
            raise ValueError('out.shape does not match result')
        if not x.dtype == out.dtype:
            raise ValueError('dtype mismatch')
    n = x.size
    wrap._unary(op, x._data, n, out._data)
    return out


def absolute(x, out=None):
    return unary(wrap.abs_op, x, out)


def cos(x, out=None):
    return unary(wrap.cos_op, x, out)


def exp(x, out=None):
    return unary(wrap.exp_op, x, out)


def fabs(x, out=None):
    return unary(wrap.abs_op, x, out)


def log(x, out=None):
    return unary(wrap.log_op, x, out)


def negative(x, out=None):
    return unary(wrap.neg_op, x, out)


def sin(x, out=None):
    return unary(wrap.sin_op, x, out)


def sqrt(x, out=None):
    return unary(wrap.sqrt_op, x, out)


def tanh(x, out=None):
    return unary(wrap.tanh_op, x, out)


def clip(a, a_min, a_max, out=None):
    out_shape = a.shape
    if out is None:
        out = base.empty(out_shape, dtype=a.dtype)
    else:
        if not out_shape == out.shape:
            raise ValueError('out.shape does not match result')
        if not a.dtype == out.dtype:
            raise ValueError('dtype mismatch')
    n = a.size
    wrap._clip(a._data, a_min, a_max, n, out._data)
    return out
