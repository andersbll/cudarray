cimport numpy as np
from .array_data cimport ArrayData
cimport elementwise


def _mul(ArrayData x1, ArrayData x2, unsigned int n, ArrayData out):
    if x1.dtype == np.dtype('float32'):
        elementwise.mul[float](<const float *>x1.dev_ptr, <const float *>x2.dev_ptr, n, <float *>out.dev_ptr)

def _mul_inplace(ArrayData x1, ArrayData x2, unsigned int n):
    if x1.dtype == np.dtype('float32'):
        elementwise.mul_inplace[float](<float *>x1.dev_ptr, <const float *>x2.dev_ptr, n)

def _mul_scalar(ArrayData x1, alpha, unsigned int n, ArrayData out):
    if x1.dtype == np.dtype('float32'):
        elementwise.mul_scalar[float](<const float *>x1.dev_ptr, <float>alpha, n, <float *>out.dev_ptr)

def _mul_scalar_inplace(ArrayData x1, alpha, unsigned int n):
    if x1.dtype == np.dtype('float32'):
        elementwise.mul_scalar_inplace[float](<float *>x1.dev_ptr, <float>alpha, n)

