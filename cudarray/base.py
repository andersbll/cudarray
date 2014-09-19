import numpy as np

from .cudarray import CUDArray


def array(object, dtype=None, copy=True):
    np_array = np.array(object)
    return CUDArray(np_array.shape, np_data=np_array)


def empty(shape, dtype=None):
    return CUDArray(shape, dtype=dtype)


def empty_like(a, dtype=None):
    if not isinstance(a, (np.ndarray, CUDArray)):
        a = np.array(a)
    return CUDArray(a.shape, dtype=a.dtype)


def ones(shape, dtype=None):
    return array(np.ones(shape, dtype=dtype))


def ones_like(a, dtype=None):
    if not isinstance(a, (np.ndarray, CUDArray)):
        a = np.array(a)
    return array(np.ones_like(a, dtype=dtype))


def zeros(shape, dtype=np.float32):
    #TODO: use fill()
    return array(np.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None):
    if not isinstance(a, (np.ndarray, CUDArray)):
        a = np.array(a)
    return array(np.zeros_like(a, dtype=dtype))
