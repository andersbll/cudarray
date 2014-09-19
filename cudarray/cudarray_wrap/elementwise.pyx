cimport numpy as np
cimport cudarray
cimport elementwise


def _mul(cudarray.CUDArray x1, cudarray.CUDArray x2, cudarray.CUDArray out):
    if not x1.dtype == x2.dtype == out.dtype:
        raise ValueError('dtype mismatch')
    if not x1.shape == x2.shape == out.shape:
        raise ValueError('shape mismatch')
    cdef int n = <int> x1.size
    if x1.dtype == np.dtype('float32'):
        elementwise.mul[float](<const float *>x1.dev_ptr, <const float *>x2.dev_ptr, n, <float *>out.dev_ptr)

def _mul_inplace(cudarray.CUDArray x1, cudarray.CUDArray x2):
    if not x1.dtype == x2.dtype:
        raise ValueError('dtype mismatch')
    if not x1.shape == x2.shape:
        raise ValueError('shape mismatch')
    cdef int n = <int> x1.size
    if x1.dtype == np.dtype('float32'):
        elementwise.mul_inplace[float](<float *>x1.dev_ptr, <const float *>x2.dev_ptr, n)

def _mul_scalar(cudarray.CUDArray x1, alpha, cudarray.CUDArray out):
    if not x1.dtype == out.dtype:
        raise ValueError('dtype mismatch')
    if not x1.shape == out.shape:
        raise ValueError('shape mismatch')
    cdef int n = <int> x1.size
    if x1.dtype == np.dtype('float32'):
        elementwise.mul_scalar[float](<const float *>x1.dev_ptr, <float>alpha, n, <float *>out.dev_ptr)

def _mul_scalar_inplace(cudarray.CUDArray x1, alpha):
    cdef int n = <int> x1.size
    if x1.dtype == np.dtype('float32'):
        elementwise.mul_scalar_inplace[float](<float *>x1.dev_ptr, <float>alpha, n)

