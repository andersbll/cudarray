import numpy as np

from .cudarray_wrap.array_data import ArrayData
import elementwise
import base


class CUDArray(object):
    def __init__(self, shape, dtype=None, np_data=None, array_data=None,
                 array_owner=None):
        self.shape = shape
        self.transposed = False
        self.isbool = False
        if dtype is None:
            if np_data is None:
                dtype = np.dtype(base.float_)
            else:
                dtype = np_data.dtype
        if np.issubdtype(dtype, float):
            dtype = np.dtype(base.float_)
        elif np.issubdtype(dtype, int):
            dtype = np.dtype(base.int_)
        elif np.issubdtype(dtype, bool):
            dtype = np.dtype(base.bool_)
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

    def __len__(self):
        return self.shape[0]

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

    def __getitem__(self, indices):
        # Standardize indices to a list of slices
        indices = _require_list(indices)
        if len(indices) > len(self.shape):
            raise IndexError('too many indices for array')
        for i, idx in enumerate(indices):
            if idx is Ellipsis:
                len_diff = self.ndim - len(indices)
                indices = indices[:i] + [slice(None)]*len_diff + indices[i:]
                return self[indices]
            elif isinstance(idx, slice):
                pass
            elif isinstance(idx, int):
                indices[i] = slice(int(idx))
            else:
                raise IndexError('only integers, slices and ellipsis are '
                                 + 'valid indices')

        # Determine view shape
        view_shape = []
        for idx, dim in zip(indices, self.shape):
            if idx.step is not None:
                raise NotImplementedError('only contiguous indices are '
                                          + 'supported')
            if idx == slice(None):
                view_shape.append(dim)
            if idx.start is not None:
                start = idx.start if idx.start >= 0 else dim - idx.start
                stop = idx.stop if idx.stop >= 0 else dim - idx.stop
                view_shape.append(stop-start)
        view_shape = tuple(view_shape)
        if len(indices) < len(self.shape):
            view_shape += self.shape[len(indices):]

        # Verify contiguous memory indexing
        is_not_full = map(lambda (idx, dim): idx.start is not None,
                          zip(indices, self.shape))
        first_full = next((i for i, full in enumerate(is_not_full)
                           if not full), len(is_not_full))
        if (any(is_not_full[first_full:])):
            raise NotImplementedError('only contiguous indices are supported')

        # Determine offset and size of view
        offset = 0
        for axis, idx in enumerate(indices):
            start = 0
            if idx.start is not None:
                start = idx.start
            elif idx.stop is not None:
                start = idx.stop
            offset += start * int(np.prod(self.shape[axis+1:]))
        size = int(np.prod(view_shape))

        # Construct view
        data_view = ArrayData(size, self.dtype, owner=self._data,
                              offset=offset)
        return CUDArray(view_shape, self.dtype, np_data=None,
                        array_data=data_view)

    def __setitem__(self, indices, c):
        view = self.__getitem__(indices)
        base.copyto(view, c)


def _require_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]
