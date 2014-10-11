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
    # TODO: use fill()
    return array(np.zeros(shape, dtype=dtype))


def zeros_like(a, dtype=None):
    if not isinstance(a, (np.ndarray, CUDArray)):
        a = np.array(a)
    return array(np.zeros_like(a, dtype=dtype))


def transpose(a):
    if a.ndim != 2:
        raise ValueError('transpose is implemented for 2D arrays only')
    a_trans = a.view()
    a_trans.shape = (a.shape[1], a.shape[0])
    a_trans.transposed = True
    return a_trans


def reshape(a, newshape):
    size = a.size
    if isinstance(newshape, int):
        newshape = (newshape,)
    newsize = np.prod(newshape)
    if size != newsize:
        if newsize < 0:
            # negative newsize means there is a -1 in newshape
            newshape = list(newshape)
            newshape[newshape.index(-1)] = -size // newsize
            newshape = tuple(newshape)
        else:
            raise ValueError('cannot reshape %s to %s' % (a.shape, newshape))
    a_reshaped = a.view()
    a_reshaped.shape = newshape
    return a_reshaped

float_ = np.float32
