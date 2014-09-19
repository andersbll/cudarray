cimport numpy as np
cimport cudarray
cimport reduction


def _sum_batched(cudarray.CUDArray a, unsigned int m, unsigned int n,
                 bool reduce_leading, cudarray.CUDArray out):
    if a.dtype == np.dtype('float32'):
        reduction.sum_batched[float](<const float *>a.dev_ptr, m, n,
                                     reduce_leading, <float *>out.dev_ptr)

