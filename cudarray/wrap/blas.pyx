cimport numpy as np
cimport blas
from .array_data cimport ArrayData, float_ptr, is_float


no_trans_op = blas.OP_NO_TRANS
trans_op = blas.OP_TRANS


def dot_(ArrayData a, ArrayData b, unsigned int n):
    if is_float(a):
        return blas.dot(float_ptr(a), float_ptr(b), n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def gemv_(ArrayData A, ArrayData x, blas.TransposeOp trans, unsigned int m,
          unsigned int n, alpha, beta, ArrayData y):
    if is_float(A):
        blas.gemv(float_ptr(A), float_ptr(x), trans, m, n, <float> alpha,
                  <float> beta, <float *>y.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(A.dtype))


def gemm_(ArrayData A, ArrayData B, blas.TransposeOp transA,
          blas.TransposeOp transB, unsigned int m, unsigned int n,
          unsigned int k, alpha, beta, ArrayData C):
    if is_float(A):
        blas.gemm(float_ptr(A), float_ptr(B), transA, transB, m, n, k,
                  <float> alpha, <float> beta, float_ptr(C))
    else:
        raise ValueError('type %s not implemented' % str(A.dtype))


cdef class BLASBatch_f:
    cdef BLASBatch[float] *ptr
    def __init__(self, ArrayData A, ArrayData B, ArrayData C, int batch_size,
                int Astride, int Bstride, int Cstride):
        self.ptr = new BLASBatch[float](float_ptr(A), float_ptr(B),
            float_ptr(C), batch_size, Astride, Bstride, Cstride)

    def __dealloc__(self):
        del self.ptr

    def gemm(self, blas.TransposeOp transA, blas.TransposeOp transB,
             unsigned int m, unsigned int n, unsigned int k, float alpha,
             float beta):
        self.ptr.gemm(transA, transB, m, n, k, alpha, beta)


cpdef blas_batch(ArrayData A, ArrayData B, ArrayData C, int batch_size,
                 int Astride, int Bstride, int Cstride):
    cdef BLASBatch[float] *ptr
    if is_float(A):
        return BLASBatch_f(A, B, C, batch_size, Astride, Bstride, Cstride)
