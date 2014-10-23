#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <cuda_runtime_api.h>
#include <cufft.h>



const char *cufftErrorEnum(cufftResult error);
#define CUFFT_CHECK(condition) { \
    cufftResult status = condition; \
    if (status != CUFFT_SUCCESS) { \
        throw std::runtime_error(cufftErrorEnum(status)); \
    } \
  }


#define CUDA_GRID_STRIDE_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; \
       i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 512

#define CUDA_THREADS_PER_BLOCK 512

#define CUDA_BLOCKS(n_threads) \
  ((n_threads + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)


namespace cudarray {

typedef int bool_t;


inline void cuda_check(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream o;
    o << file << ":" << line << ": " << cudaGetErrorString(status);
    throw std::runtime_error(o.str());
  }
}

#define CUDA_CHECK(status) { cuda_check((status), __FILE__, __LINE__); }

#ifdef CUDA_SYNC
#define CUDA_CHECK_LAST_ERROR \
  CUDA_CHECK(cudaPeekAtLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize());
#else
#define CUDA_CHECK_LAST_ERROR \
  CUDA_CHECK(cudaPeekAtLastError());
#endif


/*
  Singleton class to handle CUDA resources.
*/
class CUDA {
public:
  inline static CUDA &instance() {
    static CUDA instance_;
    return instance_;
  }

  inline static void *buffer(unsigned int size, unsigned int idx=0) {
    if (instance().buffer_sizes[idx] < size) {
      if (instance().buffers[idx]) {
        CUDA_CHECK(cudaFree(instance().buffers[idx]));
      }
      instance().buffer_sizes[idx] = size;
      CUDA_CHECK(cudaMalloc(&instance().buffers[idx], size));
    }
    return instance().buffers[idx];
  }


private:
  void *buffers[32];
  unsigned int buffer_sizes[32];

  CUDA() {
    for(int i = 0; i < 32; i++) {
        buffers[i] = NULL;
        buffer_sizes[i] = 0;
    }
    buffer_sizes[0] = 99999999;
    CUDA_CHECK(cudaMalloc(&buffers[0], buffer_sizes[0]));
  }

  ~CUDA() {
//    cudaDeviceReset();
  }

  CUDA(CUDA const&);

  void operator=(CUDA const&);
};

}


#endif  // COMMON_HPP_
