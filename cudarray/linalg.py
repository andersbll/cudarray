import numpy as np

from .cudarray_wrap import blas
from . import cudarray


def matmul_shape(a_shape, b_shape):
    a_ndim = len(a_shape)
    b_ndim = len(b_shape)
    if a_ndim == 1 and b_ndim == 2:
        if a_shape[0] != b_shape[0]:
            raise ValueError('shape mismatch')
        return (b_shape[1],)
    elif a_ndim == 2 and b_ndim == 1:
        if a_shape[1] != b_shape[0]:
            raise ValueError('shape mismatch')
        return (a_shape[0],)
    elif a_ndim == 2 and b_ndim == 2:
        if a_shape[1] != b_shape[0]:
            raise ValueError('shape mismatch')
        return (a_shape[0], b_shape[1])
    else:
        raise ValueError('only 1D and 2D arrays are supported')


def inner(a, b):
    if a.dtype != b.dtype:
        raise ValueError('dtype mismatch')
    if not a.ndim == b.ndim == 1:
            raise ValueError('shape mismatch')
    if a.size != b.size:
        raise ValueError('size mismatch')
    return blas.dot_(a._data, b._data, a.size)


def dot(a, b, out=None):
    if a.ndim == b.ndim == 1:
        return inner(a, b)

    if a.dtype != b.dtype:
        raise ValueError('dtype mismatch')

    out_shape = matmul_shape(a.shape, b.shape)
    if out is None:
        out = cudarray.empty(out_shape, dtype=a.dtype)
    else:
        if out_shape != out.shape:
            raise ValueError('out.shape does not match result')
        if a.dtype != out.dtype:
            raise ValueError('dtype mismatch')

    if a.ndim == b.ndim == 2:
        m, k = a.shape[:2]
        n = b.shape[1]
        transA = blas.trans_op if a.transposed else blas.no_trans_op
        transB = blas.trans_op if b.transposed else blas.no_trans_op
        blas.gemm_(a._data, b._data, transA, transB, m, n, k, 1.0, 0.0,
                   out._data)
    elif a.ndim == 2 and b.ndim == 1:
        m, n = a.shape
        trans = blas.trans_op if a.transposed else blas.no_trans_op
        blas.gemv_(a._data, b._data, trans, m, n, 1.0, 0.0, out._data)
    elif a.ndim == 1 and b.ndim == 2:
        n, m = b.shape
        trans = blas.no_trans_op if b.transposed else blas.trans_op
        blas.gemv_(b._data, a._data, trans, m, n, 1.0, 0.0, out._data)
    else:
        raise ValueError('invalid array dimensionality')
    return out
