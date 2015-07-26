import numpy as np
cimport numpy as np
from .cudart cimport *
from .array_data cimport ArrayData


cdef class ArrayData:
    def __init__(self, size_t size, np.dtype dtype, np.ndarray np_data=None,
                 ArrayData owner=None, unsigned int offset=0):
        self.size = size
        self.dtype = dtype
        self.nbytes = size*dtype.itemsize
        self.owner = owner
        self.offset = offset
        if owner is None:
            cudaCheck(cudaMalloc(&self.dev_ptr, self.nbytes))
        else:
            self.dev_ptr = (<char *> owner.dev_ptr) + offset*dtype.itemsize
        if np_data is not None:
            cudaCheck(cudaMemcpyAsync(self.dev_ptr, np.PyArray_DATA(np_data),
                                      self.nbytes, cudaMemcpyHostToDevice))

    def to_numpy(self, np_array):
        cudaCheck(cudaMemcpy(np.PyArray_DATA(np_array), self.dev_ptr,
                             self.nbytes, cudaMemcpyDeviceToHost))
        return np_array

    def __dealloc__(self):
        if self.owner is None:
            cudaFree(self.dev_ptr)

    def __reduce__(self):
        if self.owner is not None:
            np_array = None
        else:
            np_array = np.empty(self.size, dtype=self.dtype)
            self.to_numpy(np_array)
        args = (self.size, self.dtype, np_array, self.owner, self.offset)
        return (ArrayData, args)

    @property
    def data(self):
        return <long int> self.dev_ptr

    @property
    def itemsize(self):
        return self.dtype.itemsize


cdef bool_t *bool_ptr(ArrayData a):
    return <bool_t *> a.dev_ptr


cdef float *float_ptr(ArrayData a):
    return <float *> a.dev_ptr


cdef int *int_ptr(ArrayData a):
    return <int *> a.dev_ptr
    

cdef bool is_int(ArrayData a):
    return a.dtype == np.dtype('int32')


cdef bool is_float(ArrayData a):
    return a.dtype == np.dtype('float32')
