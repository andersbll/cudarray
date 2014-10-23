from libc.stdlib cimport malloc, free
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
#    def __init__(self, ArrayData A, ArrayData B, ArrayData C, int batch_size,
#                int Astride, int Bstride, int Cstride):
#        self.ptr = new BLASBatch[float](
#            <const float *>A.dev_ptr, <const float *>B.dev_ptr,
#            <float *>C.dev_ptr, batch_size, Astride, Bstride, Cstride)
    def __cinit__(self, const float **As, const float **Bs, float **Cs,
                  int batch_size):
        self.ptr = new BLASBatch[float](As, Bs, Cs, batch_size)

    def __dealloc__(self):
        del self.ptr

#    def gemv(self, blas.TransposeOp trans, unsigned int m, unsigned int n,
#             float alpha, float beta):
#        self.ptr.gemv(trans, m, n, alpha, beta)

    def gemm(self, blas.TransposeOp transA, blas.TransposeOp transB,
             unsigned int m, unsigned int n, unsigned int k, float alpha,
             float beta):
        self.ptr.gemm(transA, transB, m, n, k, alpha, beta)


#cdef blas_batch(void **As, void **Bs, void **C, int batch_size):
#    cdef BLASBatch[float] *ptr
#    if A.dtype == np.dtype('float32'):
#        return BLASBatch_f(As, Bs, Cs, batch_size)

#cpdef blas_batch(ArrayData A, ArrayData B, ArrayData C, int batch_size,
#                 int Astride, int Bstride, int Cstride):
#    cdef BLASBatch[float] *ptr
#    if A.dtype == np.dtype('float32'):
#        return BLASBatch_f(A, B, C, batch_size, Astride, Bstride, Cstride)

cpdef blas_batch(list A, list B, list C):
#    cdef BLASBatch[float] *ptr
    cdef int batch_size = len(A)
    cdef void **As = <void **> malloc(batch_size * sizeof(void *))
    cdef void **Bs = <void **> malloc(batch_size * sizeof(void *))
    cdef void **Cs = <void **> malloc(batch_size * sizeof(void *))
    for i, array in enumerate(A):
         As[i] = (<ArrayData> array).dev_ptr
    for i, array in enumerate(B):
         Bs[i] = (<ArrayData> array).dev_ptr
    for i, array in enumerate(C):
         Cs[i] = (<ArrayData> array).dev_ptr
    if A[0].dtype == np.dtype('float32'):
        return BLASBatch_f(<const float **> As, <const float **> Bs,
                           <float **> Cs, batch_size)
    free(As)
    free(Bs)
    free(Cs)
#        return BLASBatch_f(A, B, C, batch_size, Astride, Bstride, Cstride)
