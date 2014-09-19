import numpy as np

from cuda_wrap import cublas

import cudarray


def dot(a, b, out=None):

    m, k = a.shape[:2]
    n = b.shape[-1]

    if out is None:
        out = cudarray.empty((m, n), a.dtype)
    else:
        if not m == out.shape[0] and n == out.shape[-1]:
            raise ValueError('out.shape does not match result')

    if not a.dtype == b.dtype == out.dtype:
        raise ValueError('dtype mismatch')

    if len(a.shape) == len(b.shape) == 2:
        cublas.gemm(a, b, out, 1.0, 0.0)
    elif len(a.shape) == 1 and len(b.shape) == 2:
        pass
    elif len(a.shape) == 1 and len(b.shape) == 2:
        pass
    else:
        raise ValueError('invalid array dimensionality')
    return out
