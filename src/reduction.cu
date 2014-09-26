#include <math_constants.h>
#include "cudarray/common.hpp"
#include "cudarray/reduction.hpp"

// TODO: Parallel reductions! See
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace cudarray {

template <typename T, ReduceOp op>
__global__ void kernel_reduce(const T *a, unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, 1) {
    if (op == MAX_OP) {
      T maxval = -CUDART_INF;
      for (unsigned int i = 0; i < n; ++i) {
        if (*a > maxval) {
          maxval = *a;
        }
        ++a;
      }
      *b = maxval;
    }
    if (op == MEAN_OP) {
      T sum = 0;
      for (unsigned int i = 0; i < n; ++i) {
        sum += *a;
        ++a;
      }
      *b = sum/n;
    }
    if (op == MIN_OP) {
      T minval = CUDART_INF;
      for (unsigned int i = 0; i < n; ++i) {
        if (*a < minval) {
          minval = *a;
        }
        ++a;
      }
      *b = minval;
    }
    if (op == SUM_OP) {
      T sum = 0;
      for (unsigned int i = 0; i < n; ++i) {
        sum += *a;
        ++a;
      }
      *b = sum;
    }
  }
}

template<typename T, ReduceOp op>
void reduce(const T *a, unsigned int n, T *b) {
  kernel_reduce<T, op><<<CUDA_BLOCKS(1), CUDA_NUM_THREADS>>>(a, n, b);
}


template<typename T>
void reduce(ReduceOp op, const T *a, unsigned int n, T *b) {
  switch (op) {
    case MAX_OP:
      reduce<T, MAX_OP>(a, n, b);
      break;
    case MEAN_OP:
      reduce<T, MEAN_OP>(a, n, b);
      break;
    case MIN_OP:
      reduce<T, MIN_OP>(a, n, b);
      break;
    case SUM_OP:
      reduce<T, SUM_OP>(a, n, b);
      break;
  }
}

template void reduce<float>(ReduceOp op, const float *a, unsigned int n,
                            float *b);




template <typename T, ReduceOp op, bool reduce_leading>
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

    if (op == MAX_OP) {
      T maxval = -CUDART_INF;
      if (reduce_leading) {
        for (unsigned int i = 0; i < m; ++i) {
          if (*a > maxval) {
            maxval = *a;
          }
          a += n;
        }
      } else {
        for (unsigned int i = 0; i < n; ++i) {
          if (*a > maxval) {
            maxval = *a;
          }
          ++a;
        }
      }
      *b = maxval;
    }
    if (op == MEAN_OP) {
      T sum = 0;
      if (reduce_leading) {
        for (unsigned int i = 0; i < m; ++i) {
          sum += *a;
          a += n;
        }
      } else {
        for (unsigned int i = 0; i < n; ++i) {
          sum += *a;
          ++a;
        }
      }
      if (reduce_leading) {
        *b = sum/m;
      } else {
        *b = sum/n;
      }
    }
    if (op == MIN_OP) {
      T minval = CUDART_INF;
      if (reduce_leading) {
        for (unsigned int i = 0; i < m; ++i) {
          if (*a < minval) {
            minval = *a;
          }
          a += n;
        }
      } else {
        for (unsigned int i = 0; i < n; ++i) {
          if (*a < minval) {
            minval = *a;
          }
          ++a;
        }
      }
      *b = minval;
    }
    if (op == SUM_OP) {
      T sum = 0;
      if (reduce_leading) {
        for (unsigned int i = 0; i < m; ++i) {
          sum += *a;
          a += n;
        }
      } else {
        for (unsigned int i = 0; i < n; ++i) {
          sum += *a;
          ++a;
        }
      }
      *b = sum;
    }
  }
}

template<typename T, ReduceOp op>
void reduce_mat(const T *a, unsigned int m, unsigned int n,
                bool reduce_leading, T *b) {
  if (reduce_leading) {
    kernel_reduce_mat<T, op, true><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
        (a, m, n, b);
  } else {
    kernel_reduce_mat<T, op, false><<<CUDA_BLOCKS(m), CUDA_NUM_THREADS>>>
        (a, m, n, b);
  }
}

template<typename T>
void reduce_mat(ReduceOp op, const T *a, unsigned int m, unsigned int n,
                bool reduce_leading, T *b) {
  switch (op) {
    case MAX_OP:
      reduce_mat<T, MAX_OP>(a, m, n, reduce_leading, b);
      break;
    case MEAN_OP:
      reduce_mat<T, MEAN_OP>(a, m, n, reduce_leading, b);
      break;
    case MIN_OP:
      reduce_mat<T, MIN_OP>(a, m, n, reduce_leading, b);
      break;
    case SUM_OP:
      reduce_mat<T, SUM_OP>(a, m, n, reduce_leading, b);
      break;
  }
}

template void reduce_mat<float>(ReduceOp op, const float *a, unsigned int m,
                                unsigned int n, bool reduce_leading, float *b);

}
