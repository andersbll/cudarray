import numpy as np
import cudarray as ca
from ..cudarray_wrap import blas
from ..linalg import matmul_shape


class Dot(object):
    def __init__(self, a, b, out=None):
        self.batch_size = a.shape[0]
        self.a = a
        self.b = b
        if a.dtype != b.dtype:
            raise ValueError('dtype mismatch')
        out_shape = (self.batch_size,) + matmul_shape(a.shape[1:], b.shape[1:])
        if out is None:
            out = base.empty(out_shape, dtype=a.dtype)
        else:
            if out_shape != out.shape:
                raise ValueError('out.shape does not match result')
            if a.dtype != out.dtype:
                raise ValueError('dtype mismatch')
        self.out = out
        a_stride = np.prod(a.shape[1:])
        b_stride = np.prod(b.shape[1:])
        out_stride = np.prod(out.shape[1:])
        self.blas_batch = blas.BLASBatch_f(
            a._data, b._data, out._data, self.batch_size, a_stride, b_stride,
            out_stride
        )
        if a.ndim == b.ndim == 3:
            m, k = a.shape[1:3]
            n = b.shape[2]

            def fun():
                self.blas_batch.gemm(blas.no_trans_op, blas.no_trans_op, m, n,
                                     k, 1.0, 0.0)
                return self.out
            self.perform = fun
        else:
            raise ValueError('invalid array dimensionality')


def dot(a, b, out=None):
    return Dot(a, b, out).perform()
