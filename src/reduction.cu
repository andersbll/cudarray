#include "cudarray/common.hpp"
#include "cudarray/reduction.hpp"


namespace cudarray {

template <typename T>
__global__ void kernel_sum(const T *a, unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, 1) {
    *b = 0;
    for (unsigned int i = 0; i < n; ++i) {
      *b += *a;
      a++;
    }
  }
}

template<typename T>
void sum(const T *a, unsigned int n, T *b) {
  kernel_sum<T><<<CUDA_BLOCKS(1), CUDA_NUM_THREADS>>>(a, n, b);
}

template void sum<float>(const float *a, unsigned int n, float *b);


template <typename T>
__global__ void kernel_sum_mat_reduce_leading(const T *a, unsigned int m,
    unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a += idx;
    b += idx;
    *b = 0;
    for (unsigned int i = 0; i < m; ++i) {
      *b += *a;
      a += n;
    }
  }
}

template <typename T>
__global__ void kernel_sum_mat_reduce_trailing(const T *a, unsigned int m,
    unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, m) {
    a += idx * n;
    b += idx;
    *b = 0;
    for (unsigned int i = 0; i < n; ++i) {
      *b += *a;
      a++;
    }
  }
}

template<typename T>
void sum_mat(const T *a, unsigned int m, unsigned int n,
    bool reduce_leading, T *b) {
  if (reduce_leading) {
    kernel_sum_mat_reduce_leading<T><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
        (a, m, n, b);
  } else {
    kernel_sum_mat_reduce_trailing<T><<<CUDA_BLOCKS(m), CUDA_NUM_THREADS>>>
        (a, m, n, b);
  }
}

template void sum_mat<float>(const float *a, unsigned int m,
    unsigned int n, bool reduce_leading, float *b);

}
