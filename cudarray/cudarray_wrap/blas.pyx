cimport numpy as np
from .array_data cimport ArrayData
cimport blas


class TransOp(object):
    no_trans = blas.OP_NO_TRANS
    trans = blas.OP_TRANS


def dot_(ArrayData a, ArrayData b, unsigned int n):
    if a.dtype == np.dtype('float32'):
        return blas.dot[float](<const float *>a.dev_ptr,
                               <const float *>b.dev_ptr, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def gemv_(ArrayData A, ArrayData x, blas.TransposeOp trans, unsigned int m,
          unsigned int n, alpha, beta, ArrayData y):
    if A.dtype == np.dtype('float32'):
        blas.gemv[float](<const float *>A.dev_ptr, <const float *>x.dev_ptr,
                         trans, m, n, <float> alpha, <float> beta,
                         <float *>y.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(A.dtype))


def gemm_(ArrayData A, ArrayData B, blas.TransposeOp transA,
          blas.TransposeOp transB, unsigned int m, unsigned int n,
          unsigned int k, alpha, beta, ArrayData C):
    if A.dtype == np.dtype('float32'):
        blas.gemm[float](<const float *>A.dev_ptr, <const float *>B.dev_ptr,
                         transA, transB, m, n, k, <float> alpha, <float> beta,
                         <float *>C.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(A.dtype))
