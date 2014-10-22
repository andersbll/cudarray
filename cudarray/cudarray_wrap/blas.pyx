cimport numpy as np
from .array_data cimport ArrayData
cimport blas


no_trans_op = blas.OP_NO_TRANS
trans_op = blas.OP_TRANS


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


cdef class BLASBatch_f:
    cdef BLASBatch[float] *ptr
    def __init__(self, ArrayData A, ArrayData B, ArrayData C, int batch_size,
                int Astride, int Bstride, int Cstride):
        self.ptr = new BLASBatch[float](
            <const float *>A.dev_ptr, <const float *>B.dev_ptr,
            <float *>C.dev_ptr, batch_size, Astride, Bstride, Cstride)

    def __dealloc__(self):
        del self.ptr

    def gemm(self, blas.TransposeOp transA, blas.TransposeOp transB,
             unsigned int m, unsigned int n, unsigned int k, float alpha,
             float beta):
        self.ptr.gemm(transA, transB, m, n, k, alpha, beta)


cpdef blas_batch(ArrayData A, ArrayData B, ArrayData C, int batch_size,
                 int Astride, int Bstride, int Cstride):
    cdef BLASBatch[float] *ptr
    if A.dtype == np.dtype('float32'):
        return BLASBatch_f(A, B, C, batch_size, Astride, Bstride, Cstride)
