import os


if 'CUDARRAY_BACKEND' not in os.environ:
    # If no backend specified, try CUDA with Numpy fall-back.
    try:
        from .cudarray import *
        from .linalg import *
        from .elementwise import *
        from .reduction import *
        print('cudarray: Using CUDA back-end.')
    except:
        from .numpy_backend import *
        print('cudarray: Using Numpy back-end.')
else:
    backend = os.getenv('CUDARRAY_BACKEND', 'numpy').lower()
    if backend == 'numpy':
        from .numpy_backend import *
    elif backend == 'cuda':
        try:
            from .cudarray import *
            from .linalg import *
            from .elementwise import *
            from .reduction import *
        except:
            print('cudarray: Failed to load CUDA back-end.')
            raise
    else:
        valid_backends = ['numpy', 'cuda']
        raise ValueError('Invalid back-end "%s" specified.' % backend
                         + ' Valid options are: ' + str(valid_backends))
