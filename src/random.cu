#include <curand.h>
#include "cudarray/common.hpp"
#include "cudarray/random.hpp"


const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}


namespace cudarray {

void seed(unsigned long long val) {
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(CURAND::generator(),
                                                  val));
}

template <>
void random_normal<float>(float *a, float mu, float sigma, unsigned int n) {
  CURAND_CHECK(curandGenerateNormal(CURAND::generator(), a, n, mu, sigma));
}


template<typename T>
__global__ void kernel_stretch(T *a, T alpha, T beta, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = alpha*a[idx] + beta;
  }
}


template <>
void random_uniform<float>(float *a, float low, float high, unsigned int n) {
  CURAND_CHECK(curandGenerateUniform(CURAND::generator(), a, n));
  if (high != 1.0 || low != 0.0) {
    float alpha = high - low;
    float beta = low;
    kernel_stretch<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, alpha, beta, n);
  }
}

}
