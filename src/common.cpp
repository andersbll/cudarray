#include <stdlib.h>
#include "cudarray/common.hpp"


const char *cublasErrorString(cublasStatus_t error){
	switch(error) {
		case CUBLAS_STATUS_SUCCESS :
			return "operation completed successfully";
		case CUBLAS_STATUS_NOT_INITIALIZED :
			return "CUBLAS library not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED :
			return "resource allocation failed";
		case CUBLAS_STATUS_INVALID_VALUE :
			return "unsupported numerical value was passed to function";
		case CUBLAS_STATUS_ARCH_MISMATCH :
			return "function requires an architectural feature absent from \
			the architecture of the device";
		case CUBLAS_STATUS_MAPPING_ERROR :
			return "access to GPU memory space failed";
		case CUBLAS_STATUS_EXECUTION_FAILED :
			return "GPU program failed to execute";
		case CUBLAS_STATUS_INTERNAL_ERROR :
			return "an internal CUBLAS operation failed";
		default :
			return "other cuBLAS error";
	}
}

const char *cufftErrorEnum(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
		default :
			return "other cuFFT error";
    }
}


inline void cudaSyncCheck(const char *msg, const char *file, const int line) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i):\nCUDA error : (%d) %s\n", file, line, (int)err,
            cudaGetErrorString(err));
    fprintf(stderr, "Message : %s\n", msg);
    cudaDeviceReset();
    exit(-1);
  }
}
