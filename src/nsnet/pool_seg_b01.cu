#include <cfloat>
#include "cudarray/common.hpp"
#include "cudarray/nsnet/pool_seg_b01.hpp"

namespace cudarray {


template <typename T>
__global__ void max_pool_seg_b01(int n_threads, const T* imgs,
    int img_h, int img_w, int n_chan, int poolout_h, int poolout_w, int win_h, int win_w,
    int pad_y, int pad_x, int stride_y, int stride_x, T* poolout, int* mask) {
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {

    int n = (idx / poolout_h / poolout_w / stride_y / stride_x / n_chan);
    int ch = (idx /stride_x/stride_y) % n_chan;

    int poolout_x = idx % poolout_w;
    int poolout_y = (idx / poolout_w) % poolout_h;

    int offset_x = (idx / (poolout_w*stride_x*n_chan)) % stride_x;
    int offset_y = (idx / (poolout_h*stride_y*stride_x*n_chan)) % stride_y;

    int img_y_start = poolout_y * stride_y - pad_y + offset_y;
    int img_x_start = poolout_x * stride_x - pad_x + offset_x;

    int img_y_end = min(img_y_start + win_h, img_h);
    int img_x_end = min(img_x_start + win_w, img_w);
    img_y_start = max(img_y_start, 0);
    img_x_start = max(img_x_start, 0);
    T maxval = -FLT_MAX;
    int maxidx = -1;

    imgs += (n * n_chan + ch) * img_h * img_w;
    for (int h = img_y_start; h < img_y_end; ++h) {
      for (int w = img_x_start; w < img_x_end; ++w) {
        if (imgs[h * img_w + w] > maxval) {
          maxidx = h * img_w + w;
          maxval = imgs[maxidx];
        }
      }
    }
    poolout[idx] = maxval;
    mask[idx] = maxidx;
  }
}

template <typename T>
void max_pool_seg_b01(const T* imgs, int n_frag, int img_h, int img_w, int n_chan, int win_h,
    int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T* poolout,
    int* mask) {
  int poolout_h = (img_h + 2*pad_y - win_h) / stride_y + 1;
  int poolout_w = (img_w + 2*pad_x - win_w) / stride_x + 1;
  //Gathering, Stencil pattern 2D Moore
  //N threads = (poolout fragments) * number of channels * poolout_h * poolout_w
  int n_threads = (n_frag * stride_y * stride_x) * n_chan * poolout_h * poolout_w;
  max_pool_seg_b01<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
    n_threads, imgs, img_h, img_w, n_chan, poolout_h, poolout_w, win_h, win_w, pad_y,
    pad_x, stride_y, stride_x, poolout, mask);
  CUDA_KERNEL_CHECK;
}

template void max_pool_seg_b01<float>(const float* imgs, int n_frag, int img_h,
    int img_w, int n_chan, int win_h, int win_w, int pad_y, int pad_x, int stride_y,
    int stride_x, float* poolout, int* mask);


/*
template <typename T>
__global__ void max_pool_seg_b01_bprob(int n_threads, const T* poolout_d,
    const int* mask, int img_h, int img_w, int poolout_h, int poolout_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    T* imgs_d) {
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int img_x = idx % img_w;
    int img_y = (idx / img_w) % img_h;
    int n = idx / img_w / img_h;
    int poolout_y_start = (img_y + pad_y < win_h)
                      ? 0 : (img_y + pad_y - win_h) / stride_y + 1;
    int poolout_y_end = min((img_y + pad_y) / stride_y + 1, poolout_h);
    int poolout_x_start = (img_x + pad_x < win_w)
                      ? 0 : (img_x + pad_x - win_w) / stride_x + 1;
    int poolout_x_end = min((img_x + pad_x) / stride_x + 1, poolout_w);
    int offset = n * poolout_h * poolout_w;
    poolout_d += offset;
    mask += offset;
    T gradient = 0;
    for (int ph = poolout_y_start; ph < poolout_y_end; ++ph) {
      for (int pw = poolout_x_start; pw < poolout_x_end; ++pw) {
        if (mask[ph * poolout_w + pw] == img_y * img_w + img_x) {
          gradient += poolout_d[ph * poolout_w + pw];
        }
      }
    }
    imgs_d[idx] = gradient;
  }
}
*/

template <typename T>
__global__ void max_pool_seg_b01_bprob(int n_threads, const T* poolout_d,
    const int* mask, int img_h, int img_w, int n_chan, int poolout_h, int poolout_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    T* imgs_d) {
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {

    int n = (idx / poolout_h / poolout_w / stride_y / stride_x / n_chan);
    int ch = (idx /stride_x/stride_y) % n_chan;
    imgs_d += (n * n_chan + ch) * img_h * img_w;
    int index = mask[idx];
    if (index > -1){
      atomicAdd(&imgs_d[index], poolout_d[idx]);
    }
  }
}

template <typename T>
void max_pool_seg_b01_bprob(const T* poolout_d, const int* mask, int n_frag,
    int img_h, int img_w, int n_chan, int win_h, int win_w, int pad_y, int pad_x,
    int stride_y, int stride_x, T* imgs_d) {
  int poolout_h = (img_h + 2*pad_y - win_h) / stride_y + 1;
  int poolout_w = (img_w + 2*pad_x - win_w) / stride_x + 1;
  //N threads = (poolout fragments) * number of channels * poolout_h * poolout_w
  //One Thread for each pixel in poolout image
  int n_threads = (n_frag * stride_y * stride_x) * n_chan * poolout_h * poolout_w;
  max_pool_seg_b01_bprob<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
    n_threads, poolout_d, mask, img_h, img_w, n_chan, poolout_h, poolout_w, win_h,
    win_w, pad_y, pad_x, stride_y, stride_x, imgs_d);
  CUDA_KERNEL_CHECK;
}

template void max_pool_seg_b01_bprob(const float* poolout_d, const int* mask,
    int n_frag, int img_h, int img_w, int n_chan, int win_h, int win_w, int pad_y,
    int pad_x, int stride_y, int stride_x, float* imgs_d);


}
