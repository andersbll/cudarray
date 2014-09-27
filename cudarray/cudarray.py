import numpy as np

from .cudarray_wrap.array_data import ArrayData
import elementwise
import base


class CUDArray(object):
    def __init__(self, shape, dtype=None, np_data=None, array_data=None):
        self.shape = shape
        self.transposed = False
        if dtype is None or dtype == np.dtype('float64'):
            dtype = np.dtype('float32')
        if np_data is not None:
            np_data = np.require(np_data, dtype=dtype, requirements='C')
        if array_data is None:
            self._data = ArrayData(self.size, dtype, np_data)
        else:
            self._data = array_data

    def __array__(self):
        np_array = np.empty(self.shape, dtype=self.dtype)
        return self._data.to_numpy(np_array)

    def __str__(self):
        return self.__array__().__str__()

    def __repr__(self):
        return self.__array__().__repr__()

    def _same_array(self, other):
        return self.data == other.data

    @property
    def data(self):
        return self._data.data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def itemsize(self):
        return self._data.dtype.itemsize

    @property
    def nbytes(self):
        return np.prod(self.shape)*self.itemsize

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def T(self):
        return base.transpose(self)

    def view(self):
        return CUDArray(self.shape, self.dtype, None, self._data)

    def __add__(self, other):
        return elementwise.add(self, other)

    def __iadd__(self, other):
        return elementwise.add(self, other, self)

    def __sub__(self, other):
        return elementwise.subtract(self, other)

    def __isub__(self, other):
        return elementwise.subtract(self, other, self)

    def __mul__(self, other):
        return elementwise.multiply(self, other)

    def __imul__(self, other):
        return elementwise.multiply(self, other, self)

    def __div__(self, other):
        return elementwise.divide(self, other)

    def __idiv__(self, other):
        return elementwise.divide(self, other, self)

    def __neg__(self):
        return elementwise.negative(self)

    def __ineg__(self):
        return elementwise.negative(self, self)
