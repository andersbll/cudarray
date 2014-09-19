cimport numpy as np

from ..cuda_wrap.cudart cimport *
from .array_data cimport ArrayData


cdef class ArrayData:
    def __init__(self, size_t size, np.dtype dtype, np.ndarray np_data=None):
        self.dtype = dtype
        self.nbytes = size*dtype.itemsize
        cudaCheck(cudaMalloc(&self.dev_ptr, self.nbytes))
        if np_data is not None:
            cudaCheck(cudaMemcpy(self.dev_ptr, np.PyArray_DATA(np_data),
                                 self.nbytes, cudaMemcpyHostToDevice))

    def to_numpy(self, np_array):
        cudaCheck(cudaMemcpy(np.PyArray_DATA(np_array), self.dev_ptr,
                             self.nbytes, cudaMemcpyDeviceToHost))
        return np_array

    def __dealloc__(self):
        cudaFree(self.dev_ptr)

    @property
    def data(self):
        return <long int> self.dev_ptr

    @property
    def itemsize(self):
        return self.dtype.itemsize
