#include <cuda_runtime_api.h>
#include "cudarray/common.hpp"
#include "cudarray/array_ops.hpp"


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

namespace cudarray {


template<typename T, unsigned int axis>
__global__ void kernel_concatenate(
    const T *a, const T *b, unsigned int d0, unsigned int d1, unsigned int d2,
    unsigned int da, unsigned int db, T *c
) {
  CUDA_GRID_STRIDE_LOOP(c_idx, d0*d1*d2) {
    unsigned int c2_idx = c_idx % d2;
    unsigned int c1_idx = (c_idx / d2) % d1;
    unsigned int c0_idx = c_idx / d2 / d1;
    unsigned int a_idx;
    unsigned int b_idx;
    bool from_a;
    if (axis == 0) {
      a_idx = (c0_idx*d1 + c1_idx)*d2 + c2_idx;
      b_idx = ((c0_idx-da)*d1 + c1_idx)*d2 + c2_idx;
      from_a = c0_idx < da;
    }
    if (axis == 1) {
      a_idx = (c0_idx*da + c1_idx)*d2 + c2_idx;
      b_idx = (c0_idx*db + (c1_idx-da))*d2 + c2_idx;
      from_a = c1_idx < da;
    }
    if (axis == 2) {
      a_idx = (c0_idx*d1 + c1_idx)*da + c2_idx;
      b_idx = (c0_idx*d1 + c1_idx)*db + (c2_idx-da);
      from_a = c2_idx < da;
    }
    c[c_idx] = from_a ? a[a_idx] : b[b_idx];
  }
}

template<typename T>
void concatenate(
    const T *a, const T *b, unsigned int axis, unsigned int d0,
    unsigned int d1, unsigned int d2, unsigned int da, unsigned int db, T *c
) {
  unsigned int n = d0*d1*d2;
  if (axis == 0) {
    kernel_concatenate<T, 0><<<cuda_blocks(n), kNumBlockThreads>>>(
      a, b, d0, d1, d2, da, db, c
    );
  } else if (axis == 1) {
    kernel_concatenate<T, 1><<<cuda_blocks(n), kNumBlockThreads>>>(
      a, b, d0, d1, d2, da, db, c
    );
  } else if (axis == 2) {
    kernel_concatenate<T, 2><<<cuda_blocks(n), kNumBlockThreads>>>(
      a, b, d0, d1, d2, da, db, c
    );
  } else {
    throw std::runtime_error("invalid axis");
  }
  CUDA_KERNEL_CHECK;
}

template void concatenate<float>(
    const float *a, const float *b, unsigned int axis, unsigned int d0,
    unsigned int d1, unsigned int d2, unsigned int da, unsigned int db,
    float *c
);
template void concatenate<int>(
    const int *a, const int *b, unsigned int axis, unsigned int d0,
    unsigned int d1, unsigned int d2, unsigned int da, unsigned int db,
    int *c
);


template<typename T, unsigned int axis>
__global__ void kernel_split(
    const T *c, unsigned int d0, unsigned int d1, unsigned int d2,
    unsigned int da, unsigned int db, T *a, T *b
) {
  CUDA_GRID_STRIDE_LOOP(c_idx, d0*d1*d2) {
    unsigned int c2_idx = c_idx % d2;
    unsigned int c1_idx = (c_idx / d2) % d1;
    unsigned int c0_idx = c_idx / d2 / d1;
    unsigned int a_idx;
    unsigned int b_idx;
    bool from_a;
    if (axis == 0) {
      a_idx = (c0_idx*d1 + c1_idx)*d2 + c2_idx;
      b_idx = ((c0_idx-da)*d1 + c1_idx)*d2 + c2_idx;
      from_a = c0_idx < da;
    }
    if (axis == 1) {
      a_idx = (c0_idx*da + c1_idx)*d2 + c2_idx;
      b_idx = (c0_idx*db + (c1_idx-da))*d2 + c2_idx;
      from_a = c1_idx < da;
    }
    if (axis == 2) {
      a_idx = (c0_idx*d1 + c1_idx)*da + c2_idx;
      b_idx = (c0_idx*d1 + c1_idx)*db + (c2_idx-da);
      from_a = c2_idx < da;
    }
    T val = c[c_idx];
    if (from_a) {
        a[a_idx] = val;
    } else {
        b[b_idx] = val;
    }
  }
}


template<typename T>
void split(
    const T *c, unsigned int axis, unsigned int d0, unsigned int d1,
    unsigned int d2, unsigned int da, unsigned int db, T *a, T *b
) {
  unsigned int n = d0*d1*d2;
  if (axis == 0) {
    kernel_split<T, 0><<<cuda_blocks(n), kNumBlockThreads>>>(
      c, d0, d1, d2, da, db, a, b
    );
  } else if (axis == 1) {
    kernel_split<T, 1><<<cuda_blocks(n), kNumBlockThreads>>>(
      c, d0, d1, d2, da, db, a, b
    );
  } else if (axis == 2) {
    kernel_split<T, 2><<<cuda_blocks(n), kNumBlockThreads>>>(
      c, d0, d1, d2, da, db, a, b
    );
  } else {
    throw std::runtime_error("invalid axis");
  }
  CUDA_KERNEL_CHECK;
}

template void split<float>(
    const float *c, unsigned int axis, unsigned int d0, unsigned int d1,
    unsigned int d2, unsigned int da, unsigned int db, float *a, float *b
);
template void split<int>(
    const int *c, unsigned int axis, unsigned int d0, unsigned int d1,
    unsigned int d2, unsigned int da, unsigned int db, int *a, int *b
);


// Adapted from
// http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
template<typename T, bool mTileMultiple, bool nTileMultiple>
__global__ void kernel_transpose(const T *a, unsigned int m, unsigned int n,
                                 T *b) {
  __shared__ T tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM; i += blockDim.y) {
    int y_ = y + i;
    if (mTileMultiple || y_ < m) {
      if (nTileMultiple || x < n) {
        tile[threadIdx.y + i][threadIdx.x] = a[y_*n + x];
      }
    }
  }
  __syncthreads();

  x = blockIdx.y * blockDim.x + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM; i += blockDim.y) {
    int y_ = y + i;
    if (nTileMultiple || y_ < n) {
      if (mTileMultiple || x < m) {
        b[y_*m + x] = tile[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}


#define ceildiv(a, b) (((a)+(b)-1)/(b))

template<typename T>
void transpose(const T *a, unsigned int m, unsigned int n, T *b) {
  dim3 blocks(ceildiv(n,TILE_DIM), ceildiv(m,TILE_DIM), 1);
  dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
  if (m % TILE_DIM) {
    if (n % TILE_DIM) {
      kernel_transpose<T, false, false><<<blocks, threads>>>(a, m, n, b);
    } else {
      kernel_transpose<T, false, true><<<blocks, threads>>>(a, m, n, b);
    }
  } else {
    if (n % TILE_DIM) {
      kernel_transpose<T, true, false><<<blocks, threads>>>(a, m, n, b);
    } else {
      kernel_transpose<T, true, true><<<blocks, threads>>>(a, m, n, b);
    }
  }
  CUDA_KERNEL_CHECK;
}

template void transpose<int>(const int *a, unsigned int m, unsigned int n,
                             int *b);
template void transpose<float>(const float *a, unsigned int m, unsigned int n,
                               float *b);


template<typename Ta, typename Tb>
__global__ void kernel_as(const Ta *a, unsigned int n, Tb *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = (Tb) a[idx];
  }
}

template<typename Ta, typename Tb>
void as(const Ta *a, unsigned int n, Tb *b) {
  kernel_as<Ta, Tb><<<cuda_blocks(n), kNumBlockThreads>>>(a, n, b);
  CUDA_KERNEL_CHECK;
}

template void as<int, float>(const int *a, unsigned int n, float *b);
template void as<float, int>(const float *a, unsigned int n, int *b);


template<typename T>
__global__ void kernel_fill(T *a, unsigned int n, T alpha) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = alpha;
  }
}

template<typename T>
void fill(T *a, unsigned int n, T alpha) {
  kernel_fill<T><<<cuda_blocks(n), kNumBlockThreads>>>(a, n, alpha);
  CUDA_KERNEL_CHECK;
}

template void fill<int>(int *a, unsigned int n, int alpha);
template void fill<float>(float *a, unsigned int n, float alpha);


template<typename T>
void copy(const T *a, unsigned int n, T *b) {
  CUDA_CHECK(cudaMemcpy(b, a, n*sizeof(T), cudaMemcpyDeviceToDevice));
}

template void copy<int>(const int *a, unsigned int n, int *b);
template void copy<float>(const float *a, unsigned int n, float *b);


template<typename T>
void to_device(const T *a, unsigned int n, T *b) {
  CUDA_CHECK(cudaMemcpy(b, a, n*sizeof(T), cudaMemcpyHostToDevice));
}

template void to_device<int>(const int *a, unsigned int n, int *b);
template void to_device<float>(const float *a, unsigned int n, float *b);


template<typename T>
void to_host(const T *a, unsigned int n, T *b) {
  CUDA_CHECK(cudaMemcpy(b, a, n*sizeof(T), cudaMemcpyDeviceToHost));
}

template void to_host<int>(const int *a, unsigned int n, int *b);
template void to_host<float>(const float *a, unsigned int n, float *b);

}
