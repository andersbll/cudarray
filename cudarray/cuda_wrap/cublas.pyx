import numpy as np
cimport numpy as np

from cublas cimport *

# XXX
#from cudarray.cudarray_wrap cimport array_data
#import cudarray.cudarray_wrap.array_data as array_data

#from ..cudarray_wrap cimport array_data
#from ..cudarray_wrap import array_data

from ..cudarray_wrap.array_data cimport ArrayData
#from ..cudarray_wrap.array_data import ArrayData


cdef cublasHandle_t handle
cublasCreate(&handle)


cdef void cublasCheck(cublasStatus_t error):
    if error == CUBLAS_STATUS_SUCCESS:
        return
    elif error ==  CUBLAS_STATUS_NOT_INITIALIZED:
        raise ValueError("CUBLAS library not initialized")
    elif error == CUBLAS_STATUS_ALLOC_FAILED:
        raise ValueError("resource allocation failed")
    elif error == CUBLAS_STATUS_INVALID_VALUE:
        raise ValueError("unsupported numerical value was passed to function")
    elif error == CUBLAS_STATUS_ARCH_MISMATCH:
        raise ValueError("function requires an architectural feature absent " +
                        "the architecture of the device")
    elif error == CUBLAS_STATUS_MAPPING_ERROR:
        raise ValueError("access to GPU memory space failed")
    elif error == CUBLAS_STATUS_EXECUTION_FAILED:
        raise ValueError("GPU program failed to execute")
    elif error == CUBLAS_STATUS_INTERNAL_ERROR:
        raise ValueError("an internal CUBLAS operation failed")
    else:
        raise ValueError("other cuBLAS error")


def axpy(ArrayData x, ArrayData y, alpha):
    cdef int n = x.size
    cdef float alpha_f = alpha
    if x.dtype == np.float32:
        cublasCheck(cublasSaxpy(handle, n, &alpha_f, <const float *> x.dev_ptr,
            1, <float *> y.dev_ptr, 1))
    else:
        raise ValueError('dtype not supported (yet)')


def scal(ArrayData x, alpha):
    cdef int n = x.size
    cdef float alpha_f = alpha
    if x.dtype == np.float32:
        cublasCheck(cublasSscal(handle, n, &alpha_f, <float *> x.dev_ptr,
            1))
    else:
        raise ValueError('dtype not supported (yet)')


def gemm(ArrayData a, ArrayData b, ArrayData c, alpha, beta):
    cdef float alpha_f = alpha
    cdef float beta_f = beta

    m, k = a.shape
    _, n = b.shape

    transa = CUBLAS_OP_N
    transb = CUBLAS_OP_N
    lda = k if transa == CUBLAS_OP_N else m
    ldb = n if transb == CUBLAS_OP_N else k
    ldc = n

    if a.dtype == np.float32:
        cublasCheck(cublasSgemm(handle, transb, transa, n, m, k, &alpha_f,
            <const float *> b.dev_ptr, ldb, <const float *> a.dev_ptr, lda,
            &beta_f, <float *> c.dev_ptr, ldc))
    else:
        raise ValueError('dtype not supported (yet)')
  

