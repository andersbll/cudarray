import numpy as np
cimport numpy as np

from .cuda_wrap.cudart cimport *
from .cuda_wrap.cudart import *
cimport cudarray


cdef class CUDArray:
    def __init__(self, np_array, transfer_data=True):
        self.shape = np_array.shape
        self.dtype = np_array.dtype
        if self.dtype == np.dtype('float64'):
            self.dtype = np.dtype('float32')
        cudaCheck(cudaMalloc(&self.dev_ptr, self.nbytes))
        if transfer_data:
            np_array = np.require(np_array, dtype=self.dtype, requirements='C')
            cudaCheck(cudaMemcpy(self.dev_ptr, np.PyArray_DATA(np_array),
                                 self.nbytes, cudaMemcpyHostToDevice))

    def __array__(self):
        np_array = np.zeros(self.shape, dtype=self.dtype)
        cudaCheck(cudaMemcpy(np.PyArray_DATA(np_array), self.dev_ptr,
                             self.nbytes, cudaMemcpyDeviceToHost))
        return np_array

    def __dealloc__(self):
        cudaFree(self.dev_ptr)

    def __str__(self):
        return self.__array__().__str__()

#    def __richcmp__(self, other, op):
#        print('woop')
#        if op == 2:
#            return self.__dict__ == other.__dict__
#        return False

    @property
    def data(self):
        return <long int> self.dev_ptr

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return np.prod(self.shape)*self.itemsize

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)


def array(ndarray):
    return CUDArray(ndarray, transfer_data=True)


def zeros(shape, dtype=np.float32):
    return array(np.zeros(shape, dtype=dtype))


def zeros_like(np_array):
    return array(np.zeros(np_array.shape, dtype=np_array.dtype))


def ones(shape, dtype=np.float32):
    return array(np.ones(shape, dtype=dtype))


class fake_ndarray(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def empty(shape, dtype=np.float32):
    ndarray = fake_ndarray(shape, dtype)
    return CUDArray(ndarray, transfer_data=False)


def empty_like(ndarray, dtype=None):
    if dtype is None:
        dtype = ndarray.dtype
    ndarray = fake_ndarray(ndarray.shape, dtype)
    return CUDArray(ndarray, transfer_data=False)
