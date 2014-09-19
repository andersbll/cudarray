#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <iostream>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cufft.h>



#define CUDA_CHECK(condition) \
  { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(error)); \
    } \
  }


const char *cublasErrorString(cublasStatus_t err);

#define CUBLAS_CHECK(condition) \
  { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(cublasErrorString(status)); \
    } \
  }


const char *cufftErrorEnum(cufftResult error);
#define CUFFT_CHECK(condition) { \
    cufftResult status = condition; \
    if (status != CUFFT_SUCCESS) { \
        throw std::runtime_error(cufftErrorEnum(status)); \
    } \
  }


inline void cudaSyncCheck(const char *msg, const char *file, const int line);

#ifdef DEBUG
#define CUDA_DEBUG_SYNC(msg)
#else
#define CUDA_DEBUG_SYNC(msg)
#endif
//#define CUDA_DEBUG_SYNC(msg)
//#define CUDA_DEBUG_SYNC(msg) cudaSyncCheck(msg, __FILE__, __LINE__)

#define CUDA_GRID_STRIDE_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 512

#define CUDA_THREADS_PER_BLOCK 512

#define CUDA_BLOCKS(n_threads) \
  ((n_threads + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)



void create_ptr_list(float ***ptrs_dev, const float *base, int num, int stride, int reps=1, int rep_stride=0);


/*
  Singleton class to handle CUDA resources.
*/
class CUDA {
public:
  inline static CUDA &instance() {
    static CUDA instance_;
    return instance_;
  }

  static void require_buffer_size(int idx, int size) {
//    std::cout << "require_buffer_size(" << idx << ", " << size << ")" << std::endl;
    if (instance().buffer_sizes[idx] < size) {
      instance().buffer_sizes[idx] = size;
      if (instance().buffers[idx]) {
        cudaFree(instance().buffers[idx]);
        CUDA_DEBUG_SYNC("Could not free buffer.");
        instance().buffers[idx] = NULL;
      }
    }
  }

  inline static void *buffer(int idx=0) {
//    std::cout << "buffer(" << idx << ")" << std::endl;
    if (!instance().buffers[idx]) {
      if (instance().buffer_sizes[idx] <= 0) {
        throw std::runtime_error("No buffer size has been specified.");
      }
      CUDA_CHECK(cudaMalloc(&instance().buffers[idx],
                            instance().buffer_sizes[idx]));
    }
    // XXX: remove this
//    CUDA_CHECK(cudaMemset(instance().buffers[idx], 0, instance().buffer_sizes[idx]));
    return instance().buffers[idx];
  }

  inline static cublasHandle_t &cublas_handle() {
    return instance().cublas_handle_;
  }

private:
  cublasHandle_t cublas_handle_;
  void *buffers[3];
  int buffer_sizes[3];

  CUDA() {
    for(int i = 0; i < 3; i++) {
        buffers[i] = NULL;
        buffer_sizes[i] = -1;
    }
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  }

  ~CUDA() {
//    if (buffer_) {
//      // This segfaults with Theano (I the CUDA runtime is already shut down
//      // at this point)
////      CUDA_CHECK(cudaFree(buffer_));
//    }
//    cudaDeviceReset();
  }

  CUDA(CUDA const&);

  void operator=(CUDA const&);
};


#endif  // COMMON_HPP_
