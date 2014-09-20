cimport numpy as np

cimport reduction
from .array_data cimport ArrayData


def _sum_mat(ArrayData a, unsigned int m, unsigned int n, bool reduce_leading,
             ArrayData out):
    if a.dtype == np.dtype('float32'):
        reduction.sum_mat[float](<const float *>a.dev_ptr, m, n,
                                 reduce_leading, <float *>out.dev_ptr)

