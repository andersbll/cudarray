import numpy as np


import cudarray_wrap.elementwise as wrap
import cudarray


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
        out = cudarray.empty(broadcast_shape(x1, x2), dtype=x1.dtype)

    if np.isscalar(x1) or np.isscalar(x2):
        if np.isscalar(x1):
            alpha = x1
            scalar = x2
        else:
            array = x1
            scalar = x2
        if array is out:
            wrap._mul_scalar_inplace(array, scalar)
        else:
            wrap._mul_scalar(array, scalar, out)
        return out

    if x1 is out:
        wrap._mul_inplace(x1, x2)
    else:
        wrap._mul(x1, x2, out)
    return out
