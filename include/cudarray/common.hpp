#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <cuda_runtime_api.h>


#define CUDA_GRID_STRIDE_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; \
       i += blockDim.x * gridDim.x)


namespace cudarray {


typedef int bool_t;

const int kNumBuffers = 32;
const int kNumBlockThreads = 512;

inline int cuda_blocks(int n_threads) {
    return (n_threads + kNumBlockThreads - 1) / kNumBlockThreads;
} 

inline void cuda_check(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream o;
    o << file << ":" << line << ": " << cudaGetErrorString(status);
    throw std::runtime_error(o.str());
  }
}

#define CUDA_CHECK(status) { cuda_check((status), __FILE__, __LINE__); }

inline void cuda_kernel_check(const char *file, int line) {
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    std::ostringstream o;
    o << file << ":" << line << ": " << cudaGetErrorString(status);
    throw std::runtime_error(o.str());
  }
}

#define CUDA_KERNEL_CHECK { cuda_kernel_check(__FILE__, __LINE__); }

inline void cuda_check_sync(const char *file, const int line) {
  cuda_check(cudaDeviceSynchronize(), file, line);
}

#define CUDA_CHECK_SYNC { cuda_check_sync(__FILE__, __LINE__); }


/*
  Singleton class to handle CUDA resources.
*/
class CUDA {
public:
  inline static CUDA &instance() {
    static CUDA instance_;
    return instance_;
  }

  /*
    Request a memory pointer to device memory
  */
  inline static void *buffer(size_t size, unsigned int idx=0) {
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
  void *buffers[kNumBuffers];
  size_t buffer_sizes[kNumBuffers];

  CUDA() {
    for(int i = 0; i < kNumBuffers; i++) {
        buffers[i] = NULL;
        buffer_sizes[i] = 0;
    }
  }

  ~CUDA() {
    for(int i = 0; i < kNumBuffers; i++) {
      if (buffers[i]) {
        CUDA_CHECK(cudaFree(instance().buffers[i]));
      }
    }
  }

  CUDA(CUDA const&);
  void operator=(CUDA const&);
};


} // cudarray

#endif  // COMMON_HPP_
