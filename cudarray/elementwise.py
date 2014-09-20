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

    error =  ValueError('operands could not be broadcast together with shapes'
                        + ' ' + str(shape1) + ' ' + str(shape2))
    
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


def multiply(x1, x2, out=None):
    if np.isscalar(x1) or np.isscalar(x2):
        if np.isscalar(x1) and np.isscalar(x2):
            # abll: this should never happen and can be removed
            return x1*x2
        if np.isscalar(x1):
            scalar = x1
            array = x2
        else:
            array = x1
            scalar = x2

        inplace = False
        out_shape = array.shape
        if out is None:
            out = base.empty(out_shape, dtype=x1.dtype)
        else:
            if not out_shape == out.shape:
                raise ValueError('out.shape does not match result')
            if not array.dtype == out.dtype:
                raise ValueError('dtype mismatch')
            if array._same_array(out):
                inplace = True
        n = array.size
        if inplace:
            wrap._mul_scalar_inplace(array._data, scalar, n)
        else:
            wrap._mul_scalar(array._data, scalar, n, out._data)
        return out

    # Create/check output array
    inplace = False
    out_shape = broadcast_shape(x1.shape, x2.shape)
    if out is None:
        out = base.empty(out_shape, dtype=x1.dtype)
    else:
        if not out_shape == out.shape:
            raise ValueError('out.shape does not match result')
        if not x1.dtype == x2.dtype == out.dtype:
            raise ValueError('dtype mismatch')
        if x1._same_array(out):
            inplace = True

    btype = broadcast_type(x1.shape, x2.shape)
    if btype == NO_BROADCAST:
        n = x1.size
        if inplace:
            wrap._mul_inplace(x1._data, x2._data, n)
        else:
            wrap._mul(x1._data, x2._data, n, out._data)

    size1 = x1.size
    size2 = x2.size
    if size1 > size2:
        m, n = size1/size2, size2
    else:
        n, m = size1, size2/size1
        x1, x2 = x2, x1
        if x1._same_array(out):
            inplace = True

    if btype == BROADCAST_TO_LEADING:
        wrap._mul_broadcast(x1._data, x2._data, m, n, True, out._data)
    elif btype == BROADCAST_TO_TRAILING:
        wrap._mul_broadcast(x1._data, x2._data, m, n, False, out._data)

    return out
