import numpy as np

from .wrap import random
from . import cudarray
from . import helpers


def seed(val=None):
    if None:
        raise ValueError('not implemented')
    random._seed(val)


def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        return np.random.normal(loc, scale, size)
    size = helpers.normalize_shape(size)
    n = helpers.prod(size)
    # cuRAND number generation requires an even number of elements.
    n = n if n % 2 == 0 else n + 1
    out = cudarray.empty(n)
    random._random_normal(out._data, loc, scale, n)
    out.shape = size
    return out


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        return np.random.uniform(low, high, size)
    size = helpers.normalize_shape(size)
    n = helpers.prod(size)
    # cuRAND number generation requires an even number of elements.
    n = n if n % 2 == 0 else n + 1
    out = cudarray.empty(n)
    random._random_uniform(out._data, low, high, n)
    out.shape = size
    return out
