import os
import logging
logger = logging.getLogger(__name__)

_gpu_id = int(os.getenv('CUDARRAY_DEVICE', '0'))
print "Cudarray GPU NR = '%s'" % _gpu_id

if 'CUDARRAY_BACKEND' not in os.environ:
    # If no backend specified, try CUDA with Numpy fall-back.
    try:
        #print "5"
        from .base import *
        #print "6"
        from .cudarray import *
        #print "7"
        from .linalg import *
        #print "8"
        from .elementwise import *
        #print "9"
        from .reduction import *
        #print "1"
        from . import random
        print "2"
        from . import nnet
        print "3"
        from . import nsnet
        print "4"
        from . import batch
        print "-1"
        wrap.cudart.initialize(_gpu_id)
        print "CUDArray: Using CUDA back-end."
        logger.info('CUDArray: Using CUDA back-end.')
    except Exception,e:
        print str(e)
        from .numpy_backend import *
        print "CUDArray: Using Numpy back-end."
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
            print "CUDArray: Using CUDA back-end."
            logger.info('CUDArray: Using CUDA back-end.')
        except:
            print "CUDArray: Failed to load CUDA back-end."
            logger.error('CUDArray: Failed to load CUDA back-end.')
            raise
    else:
        valid_backends = ['numpy', 'cuda']
        raise ValueError('Invalid back-end "%s" specified.' % backend
                         + ' Valid options are: ' + str(valid_backends))
