import os

_gpu_id = int(os.getenv('CUDARRAY_DEVICE', '0'))

_backend = os.getenv('CUDARRAY_BACKEND')
if _backend is not None:
    _force_backend = True
    _backend = _backend.lower()
    _valid_backends = ['numpy', 'cuda']
    if _backend not in _valid_backends:
        raise RuntimeError('Invalid back-end "%s" specified. Valid options '
                           'are: %s' % (_backend, _valid_backends))
else:
    # If no back-end specified, try CUDA with NumPy fall-back.
    _force_backend = False
    _backend = 'cuda'

if _backend == 'cuda':
    try:
        from .cudarray import *
        from .base import *
        from .linalg import *
        from .elementwise import *
        from .reduction import *
        from . import random
        from . import nnet
        from . import batch
        from . import extra
        wrap.cudart.initialize(_gpu_id)
    except:
        if _force_backend:
            print('CUDArray: Failed to load CUDA back-end.')
            raise
        else:
            print('CUDArray: CUDA back-end not available, using NumPy.')
            # Try NumPy instead
            _backend = 'numpy'

if _backend == 'numpy':
    from .numpy_backend import *


__version__ = '0.1.dev'
