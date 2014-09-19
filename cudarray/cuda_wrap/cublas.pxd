cdef extern from "cublas_v2.h":
    struct cublasContext:
        pass
    ctypedef cublasContext *cublasHandle_t

    cublasStatus_t cublasCreate(cublasHandle_t *handle)
    ctypedef enum cublasStatus_t:
        CUBLAS_STATUS_SUCCESS
        CUBLAS_STATUS_NOT_INITIALIZED
        CUBLAS_STATUS_ALLOC_FAILED
        CUBLAS_STATUS_INVALID_VALUE
        CUBLAS_STATUS_ARCH_MISMATCH
        CUBLAS_STATUS_MAPPING_ERROR
        CUBLAS_STATUS_EXECUTION_FAILED
        CUBLAS_STATUS_INTERNAL_ERROR

    ctypedef enum cublasOperation_t:
        CUBLAS_OP_N  
        CUBLAS_OP_T  
        CUBLAS_OP_C

    cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
        const float *alpha, const float *x, int incx, float *y, int incy)

    cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx)

    cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
        cublasOperation_t transb, int m, int n, int k, const float *alpha,
        const float *A, int lda, const float *B, int ldb, const float *beta,
        float *C, int ldc)
