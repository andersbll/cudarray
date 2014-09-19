import numpy as np


import cudarray_wrap.elementwise as wrap
import base


def broadcast_shape(x1, x2):
    if np.isscalar(x1):
        return x2.shape
    elif np.isscalar(x2):
        return x1.shape
    else:
        # TODO: figure out broadcasting rules
        return x1.shape


def multiply(x1, x2, out=None):
    if np.isscalar(x1) and np.isscalar(x2):
        return x1*x2

    if out is None:
        out = base.empty(broadcast_shape(x1, x2), dtype=x1.dtype)

    if np.isscalar(x1) or np.isscalar(x2):
        if np.isscalar(x1):
            alpha = x1
            scalar = x2
        else:
            array = x1
            scalar = x2
        n = array.size
        if array is out:
            wrap._mul_scalar_inplace(array._data, scalar, n)
        else:
            wrap._mul_scalar(array._data, scalar, n, out._data)
        return out

    n = x1.size
    if x1 is out:
        wrap._mul_inplace(x1._data, x2._data, n)
    else:
        wrap._mul(x1._data, x2._data, n, out._data)
    return out
