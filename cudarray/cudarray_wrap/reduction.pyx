cimport numpy as np

cimport reduction
from .array_data cimport ArrayData


max_op = MAX_OP
mean_op = MEAN_OP
min_op = MIN_OP
sum_op = SUM_OP


def _reduce(ReduceOp op, ArrayData a, unsigned int n, ArrayData out):
    if a.dtype == np.dtype('float32'):
        reduction.reduce[float](op, <const float *>a.dev_ptr, n,
                                <float *>out.dev_ptr)
    elif a.dtype == np.dtype('int32'):
        reduction.reduce[int](op, <const int *>a.dev_ptr, n,
                                <int *>out.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _reduce_mat(ReduceOp op, ArrayData a, unsigned int m, unsigned int n,
                bool reduce_leading, ArrayData out):
    if a.dtype == np.dtype('float32'):
        reduction.reduce_mat[float](op, <const float *>a.dev_ptr, m, n,
                                    reduce_leading, <float *>out.dev_ptr)
    elif a.dtype == np.dtype('int32'):
        reduction.reduce_mat[int](op, <const int *>a.dev_ptr, m, n,
                                    reduce_leading, <int *>out.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))
