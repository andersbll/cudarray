import os
import logging
logger = logging.getLogger(__name__)

_gpu_id = int(os.getenv('CUDARRAY_DEVICE', '0'))

if 'CUDARRAY_BACKEND' not in os.environ:
    # If no backend specified, try CUDA with Numpy fall-back.
    try:
        from .base import *
        from .cudarray import *
        from .linalg import *
        from .elementwise import *
        from .reduction import *
        from . import random
        from . import nnet
        from . import nsnet
        from . import batch
        wrap.cudart.initialize(_gpu_id)
        logger.info('CUDArray: Using CUDA back-end.')
    except:
        from .numpy_backend import *
        logger.info('CUDArray: Using Numpy back-end.')
else:
    backend = os.getenv('CUDARRAY_BACKEND', 'numpy').lower()
    if backend == 'numpy':
        from .numpy_backend import *
    elif backend == 'cuda':
        try:
            from .base import *
            from .cudarray import *
            from .linalg import *
            from .elementwise import *
            from .reduction import *
            from . import random
            from . import nnet
            from . import nsnet
            from . import batch
            wrap.cudart.initialize(_gpu_id)
        except:
            logger.error('CUDArray: Failed to load CUDA back-end.')
            raise
    else:
        valid_backends = ['numpy', 'cuda']
        raise ValueError('Invalid back-end "%s" specified.' % backend
                         + ' Valid options are: ' + str(valid_backends))
