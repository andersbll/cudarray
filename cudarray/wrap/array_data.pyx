cimport numpy as np
from .cudart cimport *
from .array_data cimport ArrayData


cdef class ArrayData:
    def __init__(self, size_t size, np.dtype dtype, np.ndarray np_data=None,
                 ArrayData owner=None, unsigned int offset=0):
        self.dtype = dtype
        self.nbytes = size*dtype.itemsize
        self.owner = owner
        if owner is None:
            cudaCheck(cudaMalloc(&self.dev_ptr, self.nbytes))
        else:
            self.dev_ptr = owner.dev_ptr + offset*dtype.itemsize
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

    @property
    def data(self):
        return <long int> self.dev_ptr

    @property
    def itemsize(self):
        return self.dtype.itemsize
