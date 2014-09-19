cimport numpy as np


cdef class CUDArray:
    cdef public tuple shape
    cdef public np.dtype dtype
    cdef void *dev_ptr
