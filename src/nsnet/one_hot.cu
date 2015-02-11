#include "cudarray/common.hpp"
#include "cudarray/nnet/one_hot.hpp"

namespace cudarray {

template <typename T>
__global__ void kernel_one_hot_encode(const int *labels, int n_classes, int n,
                                      T *out) {
  CUDA_GRID_STRIDE_LOOP(idx, n*n_classes) {
    int class_idx = idx % n_classes;
    int label_idx = idx / n_classes;
    out[idx] = labels[label_idx] == class_idx ? 1.0 : 0.0;
  }
}

template <typename T>
void one_hot_encode(const int *labels, int n_classes, int n, T *out) {
  kernel_one_hot_encode<<<cuda_blocks(n*n_classes), kNumBlockThreads>>>(
      labels, n_classes, n, out);
  CUDA_KERNEL_CHECK;
}

template void one_hot_encode(const int *labels, int n_classes, int n,
                             float *out);

}
