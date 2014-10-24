#include <curand.h>
#include "cudarray/common.hpp"
#include "cudarray/random.hpp"


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
    kernel_stretch<<<cuda_blocks(n), kNumBlockThreads>>>(a, alpha, beta, n);
  }
}


const char* curand_message(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS:
      return "No errors.";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "Header file and linked library version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR:
      return "Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "Length requested is not a multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "GPU does not have double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "Preexisting failure on library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "Initialization of CUDA failed.";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "Architecture mismatch, GPU does not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "Internal library error.";
    default:
      throw std::runtime_error("invalid curandStatus_t");
  }
}


}
