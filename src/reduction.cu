#include <climits>
#include <cfloat>
#include <math_constants.h>
#include "cudarray/common.hpp"
#include "cudarray/reduction.hpp"
#include "cudarray/elementwise.hpp"

// The parallel reductions below are heavily based on
// http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// and http://cudpp.github.io/

// TODO: parallelize reduce_to_int() and reduce_mat() à la reduce()

namespace cudarray {

template <typename T>
struct SharedMemory {
    __device__ T* pointer() const;
};
template <>
__device__ inline int *SharedMemory<int>::pointer() const {
    extern __shared__ int s_int[];
    return s_int;
}
template <>
__device__ inline float *SharedMemory<float>::pointer() const {
    extern __shared__ float s_float[];
    return s_float;
}


template <typename T>
struct MaxOp {
  __device__ T identity() const;
  __device__ T operator()(const T a, const T b) {
    return max(a, b);
  }
};
template <>
__device__ inline int MaxOp<int>::identity() const {
  return INT_MIN;
}
template <>
__device__ inline float MaxOp<float>::identity() const {
  return -FLT_MAX;
}

template <typename T>
struct MinOp {
  __device__ T identity() const;
  __device__ T operator()(const T a, const T b) {
    return min(a, b);
  }
};
template <>
__device__ inline int MinOp<int>::identity() const {
  return INT_MAX;
}
template <>
__device__ inline float MinOp<float>::identity() const {
  return FLT_MAX;
}

template <typename T>
struct MulOp {
  __device__ T identity() {
    return (T) 1;
  }
  __device__ T operator()(const T a, const T b) {
    return a + b;
  }
};

template <typename T>
struct AddOp {
  __device__ T identity() {
    return (T) 0;
  }
  __device__ T operator()(const T a, const T b) {
    return a + b;
  }
};


template <typename T, typename Op, unsigned int block_size>
__global__ void reduce(const T *a, unsigned int n, T *b) {
  Op op;
  if (block_size == 1) {
    if (n == 1) {
      b[0] = a[0];
    } else if (n == 2) {
      b[0] = op(a[0], a[1]);
    }
  } else {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(block_size*2) + threadIdx.x;
    unsigned int gridSize = block_size*2*gridDim.x;

    SharedMemory<T> smem;
    volatile T* sdata = smem.pointer();
    T reduced = op.identity();

    // Reduce multiple elements per thread.
    while (i < n) {
      reduced = op(reduced, a[i]);
      // Check array bounds
      if (i + block_size < n) {
        reduced = op(reduced, a[i+block_size]);
      }
      i += gridSize;
    }

    // Reduce in shared memory
    sdata[tid] = reduced;
    __syncthreads();
    if (block_size >= 512) {
      if (tid < 256) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 256]);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 128]);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid <  64) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 64]);
      }
      __syncthreads();
    }
    // No need to sync threads in the same warp
    if (tid < 32) {
      if (block_size >= 64) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 32]);
      }
      if (block_size >= 32) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 16]);
      }
      if (block_size >= 16) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 8]);
      }
      if (block_size >= 8) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 4]);
      }
      if (block_size >= 4) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 2]);
      }
      if (block_size >= 2) {
        sdata[tid] = reduced = op(reduced, sdata[tid + 1]);
      }
    }

    // Write reduced block back to global memory
    if (tid == 0) {
      b[blockIdx.x] = sdata[0];
    }
  }
}

const unsigned int max_blocks = 64;
const unsigned int reduce_cta_size =  256;
inline unsigned int ceil_pow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}


unsigned int n_reduce_blocks(unsigned int n) {
  return min(max_blocks, (n + (2*reduce_cta_size - 1)) / (2*reduce_cta_size));
}

unsigned int n_reduce_threads(unsigned int n) {
  return n > 2 * reduce_cta_size ? reduce_cta_size : max(1, ceil_pow2(n) / 2);
}

template <typename T, typename Op>
void reduce_blocks(const T *a, unsigned int n, T *b) {
  unsigned int n_threads = n_reduce_threads(n);
  dim3 block(n_threads, 1, 1);

  unsigned int n_blocks = n_reduce_blocks(n);
  dim3 grid(n_blocks, 1, 1);
  int smem_size = reduce_cta_size * sizeof(T);

  switch (block.x) {
    case 512:
      reduce<T, Op, 512><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 256:
      reduce<T, Op, 256><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 128:
      reduce<T, Op, 128><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 64:
      reduce<T, Op, 64><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 32:
      reduce<T, Op, 32><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 16:
      reduce<T, Op, 16><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 8:
      reduce<T, Op, 8><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 4:
      reduce<T, Op, 4><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 2:
      reduce<T, Op, 2><<<grid, block, smem_size>>>(a, n, b);
      break;
    case 1:
      reduce<T, Op, 1><<<grid, block, smem_size>>>(a, n, b);
      break;
  }
}

template <typename T, typename Op>
void reduce(const T *a, unsigned int n, T *b) {
  unsigned int n_blocks = n_reduce_blocks(n);
  if (n_blocks > 1) {
    T *buf = (T *) CUDA::buffer(n_blocks*sizeof(T));
    reduce_blocks<T, Op>(a, n, buf);
    reduce_blocks<T, Op>(buf, n_blocks, b);
  } else {
    reduce_blocks<T, Op>(a, n, b);
  }
}

template<typename T>
void reduce(ReduceOp op, const T *a, unsigned int n, T *b) {
  switch (op) {
    case MAX_OP:
      reduce<T, MaxOp<T> >(a, n, b);
      break;
    case MEAN_OP:
      reduce<T, AddOp<T> >(a, n, b);
      binary_scalar(DIV_OP, b, (T) n, 1, b);
      break;
    case MIN_OP:
      reduce<T, MinOp<T> >(a, n, b);
      break;
    case SUM_OP:
      reduce<T, AddOp<T> >(a, n, b);
      break;
  }
}

template void reduce<float>(ReduceOp op, const float *a, unsigned int n,
                            float *b);
template void reduce<int>(ReduceOp op, const int *a, unsigned int n,
                            int *b);



#define REDUCE_OP(name, ident_f, ident_i, reduce_op, scale_op, select_op) \
template <typename Tb> \
struct name; \
template <> \
struct name<float> { \
  __device__ inline static float identity() { \
    return ident_f; \
  } \
  template <typename Ta, typename Tb> \
  __device__ inline static void reduce(volatile Ta a, volatile int idx, \
                                       volatile Tb &b, volatile int &b_idx) { \
    reduce_op; \
  } \
  template <typename Tb> \
  __device__ inline static void scale(volatile Tb &b, volatile float n) { \
    scale_op; \
  } \
  template <typename Ta, typename Tb> \
  __device__ inline static void select(volatile Tb &b, volatile Ta a, \
                                       volatile int idx) { \
    select_op; \
  } \
}; \
template <> \
struct name<int> { \
  __device__ inline static int identity() { \
    return ident_i; \
  } \
  template <typename Ta, typename Tb> \
  __device__ inline static void reduce(volatile Ta a, volatile int idx, \
                                       volatile Tb &b, volatile int &b_idx) { \
    reduce_op; \
  } \
  template <typename Tb> \
  __device__ inline static void scale(volatile Tb &b, volatile float n) { \
    scale_op; \
  } \
  template <typename Ta, typename Tb> \
  __device__ inline static void select(volatile Tb &b, volatile Ta a, \
                                       volatile int idx) { \
    select_op; \
  } \
};

REDUCE_OP(max_op, -FLT_MAX, INT_MIN, if (a > b) b = a, , b = a)
REDUCE_OP(mean_op, 0.0f, 0, b += a, b /= n, b = a)
REDUCE_OP(min_op, FLT_MAX, INT_MAX, if (a < b) b = a, , b = a)
REDUCE_OP(sum_op, 0.0f, 0, b += a, , b = a)
REDUCE_OP(argmax_op, -FLT_MAX, INT_MIN, if (a > b) {b = a; b_idx=idx;}, , b = idx)
REDUCE_OP(argmin_op, FLT_MAX, INT_MAX, if (a < b) {b = a; b_idx=idx;}, , b = idx)



template <typename Ta, typename Tb, typename Op>
__global__ void kernel_reduce(const Ta *a, unsigned int n, Tb *b) {
  CUDA_GRID_STRIDE_LOOP(idx, 1) {
    Ta a_ = Op::identity();
    int idx_ = 0;
    for (unsigned int i = 0; i < n; ++i) {
      Op::reduce(*a, i, a_, idx_);
      ++a;
    }
    Op::scale(a_, n);
    Op::select(*b, a_, idx_);
  }
}


template <typename Ta, typename Tb, typename Op>
void reduce(const Ta *a, unsigned int n, Tb *b) {
  kernel_reduce<Ta, Tb, Op><<<cuda_blocks(1), kNumBlockThreads>>>(a, n, b);
}




template<typename T>
void reduce_to_int(ReduceToIntOp op, const T *a, unsigned int n, int *b) {
  switch (op) {
    case ARGMAX_OP:
      reduce<T, int, argmax_op<T> >(a, n, b);
      break;
    case ARGMIN_OP:
      reduce<T, int, argmin_op<T> >(a, n, b);
      break;
  }
}

template void reduce_to_int<float>(ReduceToIntOp op, const float *a,
                                   unsigned int n, int *b);
template void reduce_to_int<int>(ReduceToIntOp op, const int *a,
                                 unsigned int n, int *b);





template <typename Ta, typename Tb, typename Op, bool reduce_leading>
__global__ void kernel_reduce_mat(const Ta *a, unsigned int m, unsigned int n,
                                  Tb *b) {
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

    Ta a_ = Op::identity();
    int idx_ = 0;
    if (reduce_leading) {
      for (unsigned int i = 0; i < m; ++i) {
        Op::reduce(*a, i, a_, idx_);
        a += n;
      }
    } else {
      for (unsigned int i = 0; i < n; ++i) {
        Op::reduce(*a, i, a_, idx_);
        ++a;
      }
    }

    if (reduce_leading) {
      Op::scale(a_, m);
    } else {
      Op::scale(a_, n);
    }
    Op::select(*b, a_, idx_);
  }
}

template<typename Ta, typename Tb, typename Op>
void reduce_mat(const Ta *a, unsigned int m, unsigned int n,
                bool reduce_leading, Tb *b) {
  if (reduce_leading) {
    kernel_reduce_mat<Ta, Tb, Op, true><<<cuda_blocks(n), kNumBlockThreads>>>
        (a, m, n, b);
  } else {
    kernel_reduce_mat<Ta, Tb, Op, false><<<cuda_blocks(m), kNumBlockThreads>>>
        (a, m, n, b);
  }
}

template<typename T>
void reduce_mat(ReduceOp op, const T *a, unsigned int m, unsigned int n,
                bool reduce_leading, T *b) {
  switch (op) {
    case MAX_OP:
      reduce_mat<T, T, max_op<T> >(a, m, n, reduce_leading, b);
      break;
    case MEAN_OP:
      reduce_mat<T, T, mean_op<T> >(a, m, n, reduce_leading, b);
      break;
    case MIN_OP:
      reduce_mat<T, T, min_op<T> >(a, m, n, reduce_leading, b);
      break;
    case SUM_OP:
      reduce_mat<T, T, sum_op<T> >(a, m, n, reduce_leading, b);
      break;
  }
}

template void reduce_mat<float>(ReduceOp op, const float *a, unsigned int m,
                                unsigned int n, bool reduce_leading, float *b);
template void reduce_mat<int>(ReduceOp op, const int *a, unsigned int m,
                                unsigned int n, bool reduce_leading, int *b);


template<typename T>
void reduce_mat_to_int(ReduceToIntOp op, const T *a, unsigned int m,
                       unsigned int n, bool reduce_leading, int *b) {
  switch (op) {
    case ARGMAX_OP:
      reduce_mat<T, int, argmax_op<T> >(a, m, n, reduce_leading, b);
      break;
    case ARGMIN_OP:
      reduce_mat<T, int, argmin_op<T> >(a, m, n, reduce_leading, b);
      break;
  }
}

template void reduce_mat_to_int<float>(ReduceToIntOp op, const float *a,
    unsigned int m, unsigned int n, bool reduce_leading, int *b);
template void reduce_mat_to_int<int>(ReduceToIntOp op, const int *a,
    unsigned int m, unsigned int n, bool reduce_leading, int *b);

}
