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


//void create_ptr_list(float ***ptrs_dev, const float *base, int num, int stride, int reps=1, int rep_stride=0) {
//  float *ptrs_host[num*reps];
//  int idx = 0;
//  for(int r = 0; r < reps; r++){
//      for(int n = 0; n < num; n++){
//        ptrs_host[idx] = (float *) base + n * stride + r * rep_stride;
////        std::cout << n << ": " << ptrs_host[idx] << "    ";
//        idx++;
//      }
//  }
////  std::cout << std::endl;
////  std::cout << num*reps*sizeof(float *) << "  " << *ptrs_dev << "  " << ptrs_host << "  " << std::endl;
//  CUDA_CHECK(cudaMemcpy(*ptrs_dev, ptrs_host, num*reps*sizeof(float *),
//                        cudaMemcpyHostToDevice));
////  CUDA_DEBUG_SYNC("create_ptr_list() failed");
//}


///*
//  Singleton class to handle CUDA resources.
//*/
//class CUDA {
//public:
//  inline static CUDA &instance() {
//    static CUDA instance_;
//    return instance_;
//  }

//  static void require_buffer_size(int idx, int size) {
////    std::cout << "require_buffer_size(" << idx << ", " << size << ")" << std::endl;
//    if (instance().buffer_sizes[idx] < size) {
//      instance().buffer_sizes[idx] = size;
//      if (instance().buffers[idx]) {
//        cudaFree(instance().buffers[idx]);
//        CUDA_DEBUG_SYNC("Could not free buffer.");
//        instance().buffers[idx] = NULL;
//      }
//    }
//  }

//  inline static void *buffer(int idx=0) {
////    std::cout << "buffer(" << idx << ")" << std::endl;
//    if (!instance().buffers[idx]) {
//      if (instance().buffer_sizes[idx] <= 0) {
//        throw std::runtime_error("No buffer size has been specified.");
//      }
//      CUDA_CHECK(cudaMalloc(&instance().buffers[idx],
//                            instance().buffer_sizes[idx]));
//    }
//    // XXX: remove this
////    CUDA_CHECK(cudaMemset(instance().buffers[idx], 0, instance().buffer_sizes[idx]));
//    return instance().buffers[idx];
//  }

//  inline static cublasHandle_t &cublas_handle() {
//    return instance().cublas_handle_;
//  }

//private:
//  cublasHandle_t cublas_handle_;
//  void *buffers[3];
//  int buffer_sizes[3];

//  CUDA() {
//    for(int i = 0; i < 3; i++) {
//        buffers[i] = NULL;
//        buffer_sizes[i] = -1;
//    }
//    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
//  }

//  ~CUDA() {
////    if (buffer_) {
////      // This segfaults with Theano (I the CUDA runtime is already shut down
////      // at this point)
//////      CUDA_CHECK(cudaFree(buffer_));
////    }
////    cudaDeviceReset();
//  }

//  CUDA(CUDA const&);

//  void operator=(CUDA const&);
//};

