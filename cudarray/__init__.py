import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s %(message)s',
)

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
        from .base import *
        from .cudarray import *
        from .linalg import *
        from .elementwise import *
        from .reduction import *
        from . import random
        from . import nnet
        from . import batch
        wrap.cudart.initialize(_gpu_id)
    except:
        if _force_backend:
            logger.error('CUDArray: Failed to load CUDA back-end.')
            raise
        else:
            # Try NumPy instead
            _backend = 'numpy'

if _backend == 'numpy':
    from .numpy_backend import *


__version__ = '0.1.dev'
