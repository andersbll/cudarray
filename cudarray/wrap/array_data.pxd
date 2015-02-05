from libcpp cimport bool
cimport numpy as np

cdef extern from 'cudarray/common.hpp' namespace 'cudarray':
    ctypedef int bool_t;

cdef class ArrayData:
    cdef public np.dtype dtype
    cdef public unsigned int nbytes
    cdef void *dev_ptr
    cdef ArrayData owner
    cdef size_t size
    cdef unsigned int offset


cdef bool_t *bool_ptr(ArrayData a)
cdef float *float_ptr(ArrayData a)
cdef int *int_ptr(ArrayData a)
cdef bool is_int(ArrayData a)
cdef bool is_float(ArrayData a)
