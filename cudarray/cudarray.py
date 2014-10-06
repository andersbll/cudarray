import numpy as np

from .cudarray_wrap.array_data import ArrayData
import elementwise
import base


class CUDArray(object):
    def __init__(self, shape, dtype=None, np_data=None, array_data=None):
        self.shape = shape
        self.transposed = False
        self.isbool = False
        if dtype is None:
            if np_data is None:
                dtype = np.dtype('float32')
            else:
                dtype = np_data.dtype
        if dtype == np.dtype('float64'):
            dtype = np.dtype('float32')
        if dtype == np.dtype('int64'):
            dtype = np.dtype('int32')
        if dtype == np.dtype('bool'):
            # TODO: figure out if bool should stay as char
            dtype = np.dtype('int32')
            self.isbool = True
        if np_data is not None:
            np_data = np.require(np_data, dtype=dtype, requirements='C')
        if array_data is None:
            self._data = ArrayData(self.size, dtype, np_data)
        else:
            self._data = array_data

    def __array__(self):
        np_array = np.empty(self.shape, dtype=self.dtype)
        self._data.to_numpy(np_array)
        if self.isbool:
            np_array = np_array.astype(np.dtype('bool'))
        return np_array

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

    def __radd__(self, other):
        return elementwise.add(self, other)

    def __iadd__(self, other):
        return elementwise.add(self, other, self)

    def __sub__(self, other):
        return elementwise.subtract(self, other)

    def __rsub__(self, other):
        return elementwise.subtract(self, other)

    def __isub__(self, other):
        return elementwise.subtract(self, other, self)

    def __mul__(self, other):
        return elementwise.multiply(self, other)

    def __rmul__(self, other):
        return elementwise.multiply(self, other)

    def __imul__(self, other):
        return elementwise.multiply(self, other, self)

    def __div__(self, other):
        return elementwise.divide(self, other)

    def __rdiv__(self, other):
        return elementwise.divide(self, other)

    def __idiv__(self, other):
        return elementwise.divide(self, other, self)

    def __pow__(self, other):
        return elementwise.power(self, other)

    def __rpow__(self, other):
        return elementwise.power(self, other)

    def __ipow__(self, other):
        return elementwise.power(self, other, self)

    def __eq__(self, other):
        return elementwise.equal(self, other)

    def __gt__(self, other):
        return elementwise.greater(self, other)

    def __ge__(self, other):
        return elementwise.greater_equal(self, other)

    def __lt__(self, other):
        return elementwise.less(self, other)

    def __le__(self, other):
        return elementwise.less_equal(self, other)

    def __ne__(self, other):
        return elementwise.not_equal(self, other)

    def __neg__(self):
        return elementwise.negative(self)

    def __ineg__(self):
        return elementwise.negative(self, self)
