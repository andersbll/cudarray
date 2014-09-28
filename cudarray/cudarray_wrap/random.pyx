cimport numpy as np

cimport random
from .array_data cimport ArrayData


def _seed(val):
    random.seed(<unsigned long long> val)


def _random_normal(ArrayData a, mu, sigma, unsigned int n):
    if a.dtype == np.dtype('float32'):
        random.random_normal[float](<float *>a.dev_ptr, <float> mu,
                                    <float> sigma, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _random_uniform(ArrayData a, low, high, unsigned int n):
    if a.dtype == np.dtype('float32'):
        random.random_uniform[float](<float *>a.dev_ptr, <float> low,
                                     <float> high, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))
