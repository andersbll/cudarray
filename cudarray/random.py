import numpy as np

import cudarray_wrap.random as wrap
import base
from .helpers import normalize_shape


def seed(val=None):
    if None:
        raise ValueError('not implemented')
    wrap._seed(val)


def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        return np.random.normal(loc, scale, size)
    size = normalize_shape(size)
    n = np.prod(size)
    # cuRAND number generation requires an even number of elements.
    n = n if n % 2 == 0 else n + 1
    out = base.empty(n)
    wrap._random_normal(out._data, loc, scale, n)
    out.shape = size
    return out


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        return np.random.uniform(low, high, size)
    size = normalize_shape(size)
    n = np.prod(size)
    # cuRAND number generation requires an even number of elements.
    n = n if n % 2 == 0 else n + 1
    out = base.empty(n)
    wrap._random_uniform(out._data, low, high, n)
    out.shape = size
    return out
