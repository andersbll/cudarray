cimport numpy as np
cimport random
from .array_data cimport ArrayData, float_ptr, is_float


def _seed(val):
    random.seed(<unsigned long long> val)


def _random_normal(ArrayData a, mu, sigma, unsigned int n):
    if is_float(a):
        random.random_normal(float_ptr(a), <float> mu, <float> sigma, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _random_uniform(ArrayData a, low, high, unsigned int n):
    if is_float(a):
        random.random_uniform(float_ptr(a), <float> low, <float> high, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))
