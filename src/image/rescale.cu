#include "cudarray/common.hpp"
#include "cudarray/image/rescale.hpp"
#include <math.h>

namespace cudarray {

template <typename T>
__global__ void kernel_rescale_bilinear(
    const T * __restrict__ imgs, float factor, int n_imgs, int img_h,
    int img_w, int scaled_h, int scaled_w, T * __restrict__ imgs_scaled) {
  CUDA_GRID_STRIDE_LOOP(idx, n_imgs*scaled_h*scaled_w) {
    int x = idx % scaled_w;
    int y = (idx / scaled_w) % scaled_h;
    int n = idx / scaled_w / scaled_h;
    float img_x = (x+0.5) / (img_w*factor) * (img_w - 1);
    float img_y = (y+0.5) / (img_h*factor) * (img_h - 1);
    int img_x0 = max((int) floor(img_x), 0);
    int img_y0 = max((int) floor(img_y), 0);
    int img_x1 = min((int) img_x+1, img_w-1);
    int img_y1 = min((int) img_y+1, img_h-1);

    T val_00 = imgs[(n*img_h + img_y0)*img_w + img_x0];
    T val_01 = imgs[(n*img_h + img_y0)*img_w + img_x1];
    T val_10 = imgs[(n*img_h + img_y1)*img_w + img_x0];
    T val_11 = imgs[(n*img_h + img_y1)*img_w + img_x1];

    float a = img_x - img_x0;
    float b = img_y - img_y0;

    T val_0 = a*val_01 + (1.0 - a)*val_00;
    T val_1 = a*val_11 + (1.0 - a)*val_10;
    T val = b*val_1 + (1.0 - b)*val_0;

    imgs_scaled[(n*scaled_h + y)*scaled_w + x] = val;
   } 
}

template <typename T>
__global__ void kernel_upsample_perforated(
    const T *imgs, int factor, int n_imgs, int img_h, int img_w,
    int scaled_h, int scaled_w, T *imgs_scaled) {
  CUDA_GRID_STRIDE_LOOP(idx, n_imgs*scaled_h*scaled_w) {
    int x = idx % scaled_w;
    int y = (idx / scaled_w) % scaled_h;
    int n = idx / scaled_w / scaled_h;
    T val;
    if (x % factor || y % factor) {
      val = 0;
    } else {
      int img_x = x / factor;
      int img_y = y / factor;
      val = imgs[(n*img_h + img_y)*img_w + img_x];
    }
    imgs_scaled[(n*scaled_h + y)*scaled_w + x] = val;
  }
}


template <typename T>
__global__ void kernel_rescale_nearest(
    const T *imgs, float factor, int n_imgs, int img_h, int img_w,
    int scaled_h, int scaled_w, T *imgs_scaled) {
  CUDA_GRID_STRIDE_LOOP(idx, n_imgs*scaled_h*scaled_w) {
    int x = idx % scaled_w;
    int y = (idx / scaled_w) % scaled_h;
    int n = idx / scaled_w / scaled_h;
    int img_x = floor(x / factor);
    int img_y = floor(y / factor);
    T val = imgs[(n*img_h + img_y)*img_w + img_x];
    imgs_scaled[(n*scaled_h + y)*scaled_w + x] = val;
  }
}


template <typename T>
void rescale(const T *imgs, float factor, SampleMethod method, int n_imgs,
             int img_h, int img_w, T *imgs_scaled) {
  if (factor <= 0) {
    throw std::runtime_error("Factor must be positive.");
  }
  int scaled_h;
  int scaled_w;
  if (factor < 1) {
    scaled_h = ceil(img_h*factor);
    scaled_w = ceil(img_w*factor);
  } else {
    scaled_h = floor(img_h*factor);
    scaled_w = floor(img_w*factor);
  }
  int n_threads = n_imgs * scaled_h * scaled_w;
  switch(method) {
  case BILINEAR_SAMPLING:
    kernel_rescale_bilinear<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
        imgs, factor, n_imgs, img_h, img_w, scaled_h, scaled_w, imgs_scaled
    );
    break;
  case NEAREST_SAMPLING:
    kernel_rescale_nearest<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
        imgs, factor, n_imgs, img_h, img_w, scaled_h, scaled_w, imgs_scaled
    );
    break;
  case PERFORATED_SAMPLING:
    if (factor < 1) {
      kernel_rescale_nearest<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
          imgs, factor, n_imgs, img_h, img_w, scaled_h, scaled_w, imgs_scaled
      );
    } else {
      if (ceilf(factor) != factor) {
        throw std::runtime_error("Factor must be integer for perforated upscaling.");
      }
      kernel_upsample_perforated<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
          imgs, factor, n_imgs, img_h, img_w, scaled_h, scaled_w, imgs_scaled
      );
    }
    break;
  default:
    throw std::runtime_error("Invalid method.");
  }
  CUDA_KERNEL_CHECK;
}


template void rescale(const float *imgs, float factor, SampleMethod method,
    int n_imgs, int img_h, int img_w, float *imgs_scaled);

}
