#include <climits>
#include <cfloat>
#include <math_constants.h>
#include "cudarray/common.hpp"
#include "cudarray/reduction.hpp"

// TODO: Parallel reductions! See
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace cudarray {

#define REDUCE_OP(name, ident_f, ident_i, reduce_op, scale_op) \
template <typename Tb> \
struct name; \
template <> \
struct name<float> { \
  const static float identity = ident_f; \
  template <typename Ta, typename Tb> \
  __device__ inline static void reduce(volatile Ta a, volatile Tb &b) { \
    reduce_op; \
  } \
  template <typename Tb> \
  __device__ inline static void scale(volatile Tb &b, volatile float n) { \
    scale_op; \
  } \
}; \
template <> \
struct name<int> { \
  const static int identity = ident_i; \
  template <typename Ta, typename Tb> \
  __device__ inline static void reduce(volatile Ta a, volatile Tb &b) { \
    reduce_op; \
  } \
  template <typename Tb> \
  __device__ inline static void scale(volatile Tb &b, volatile float n) { \
    scale_op; \
  } \
};

REDUCE_OP(max_op, -FLT_MAX, INT_MIN, if (a > b) b = a, )
REDUCE_OP(mean_op, 0.0f, 0, b += a, b /= n)
REDUCE_OP(min_op, FLT_MAX, INT_MAX, if (a < b) b = a, )
REDUCE_OP(sum_op, 0.0f, 0, b += a, )



template <typename T, typename Op>
__global__ void kernel_reduce(const T *a, unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, 1) {
    T b_ = Op::identity;
    for (unsigned int i = 0; i < n; ++i) {
      Op::reduce(*a, b_);
      ++a;
    }
    Op::scale(b_, n);
    *b = b_;
  }
}


template<typename T, typename Op>
void reduce(const T *a, unsigned int n, T *b) {
  kernel_reduce<T, Op><<<CUDA_BLOCKS(1), CUDA_NUM_THREADS>>>(a, n, b);
}


template<typename T>
void reduce(ReduceOp op, const T *a, unsigned int n, T *b) {
  switch (op) {
    case MAX_OP:
      reduce<T, max_op<T> >(a, n, b);
      break;
    case MEAN_OP:
      reduce<T, mean_op<T> >(a, n, b);
      break;
    case MIN_OP:
      reduce<T, min_op<T> >(a, n, b);
      break;
    case SUM_OP:
      reduce<T, sum_op<T> >(a, n, b);
      break;
  }
}

template void reduce<float>(ReduceOp op, const float *a, unsigned int n,
                            float *b);
template void reduce<int>(ReduceOp op, const int *a, unsigned int n,
                            int *b);




template <typename T, typename Op, bool reduce_leading>
__global__ void kernel_reduce_mat(const T *a, unsigned int m, unsigned int n,
                                  T *b) {
  unsigned int n_threads;
  if (reduce_leading) {
    n_threads = n;
  } else {
    n_threads = m;
  }

  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    if (reduce_leading) {
      a += idx;
      b += idx;
    } else {
      a += idx * n;
      b += idx;
    }

    T b_ = Op::identity;
    if (reduce_leading) {
      for (unsigned int i = 0; i < m; ++i) {
        Op::reduce(*a, b_);
        a += n;
      }
    } else {
      for (unsigned int i = 0; i < n; ++i) {
        Op::reduce(*a, b_);
        ++a;
      }
    }

    if (reduce_leading) {
      Op::scale(b_, m);
    } else {
      Op::scale(b_, n);
    }
    *b = b_;
  }
}

template<typename T, typename Op>
void reduce_mat(const T *a, unsigned int m, unsigned int n,
                bool reduce_leading, T *b) {
  if (reduce_leading) {
    kernel_reduce_mat<T, Op, true><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
        (a, m, n, b);
  } else {
    kernel_reduce_mat<T, Op, false><<<CUDA_BLOCKS(m), CUDA_NUM_THREADS>>>
        (a, m, n, b);
  }
}

template<typename T>
void reduce_mat(ReduceOp op, const T *a, unsigned int m, unsigned int n,
                bool reduce_leading, T *b) {
  switch (op) {
    case MAX_OP:
      reduce_mat<T, max_op<T> >(a, m, n, reduce_leading, b);
      break;
    case MEAN_OP:
      reduce_mat<T, mean_op<T> >(a, m, n, reduce_leading, b);
      break;
    case MIN_OP:
      reduce_mat<T, min_op<T> >(a, m, n, reduce_leading, b);
      break;
    case SUM_OP:
      reduce_mat<T, sum_op<T> >(a, m, n, reduce_leading, b);
      break;
  }
}

template void reduce_mat<float>(ReduceOp op, const float *a, unsigned int m,
                                unsigned int n, bool reduce_leading, float *b);
template void reduce_mat<int>(ReduceOp op, const int *a, unsigned int m,
                                unsigned int n, bool reduce_leading, int *b);

}
