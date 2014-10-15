cimport numpy as np


cdef class ArrayData:
    cdef public np.dtype dtype
    cdef public unsigned int nbytes
    cdef void *dev_ptr
    cdef ArrayData owner
