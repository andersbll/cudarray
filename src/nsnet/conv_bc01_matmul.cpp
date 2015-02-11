#include "cudarray/common.hpp"
#include "cudarray/blas.hpp"
#include "cudarray/nnet/conv_bc01_matmul.hpp"
#include "cudarray/image/img2win.hpp"

// The following convolution operations by matrix multiplication are heavily
// inspired by those from Caffe, http://caffe.berkeleyvision.org/

namespace cudarray {

template <typename T>
void conv_bc01_matmul(const T *imgs, const T *filters, int n_imgs,
    int n_channels, int n_filters, int img_h, int img_w, int filter_h,
    int filter_w, int pad_y, int pad_x, int stride_y, int stride_x,
    T *convout) {
  int convout_h = (img_h + 2 * pad_y - filter_h) / stride_y + 1;
  int convout_w = (img_w + 2 * pad_x - filter_w) / stride_x + 1;
  int win_size = filter_h * filter_w;
  T *buffer = (T *) CUDA::buffer(sizeof(float) * n_channels * win_size
                                 * convout_h * convout_w);
  int m = n_filters;
  int k = n_channels*win_size;
  int n = convout_h * convout_w;
  for (int i = 0; i < n_imgs; ++i) {
    const T *img = imgs + i * n_channels * img_h * img_w;
    img2win(img, n_channels, img_h, img_w, filter_h, filter_w, pad_y, pad_x,
            stride_y, stride_x, buffer);
    T *convout_img = convout + i * n_filters * convout_h * convout_w;
    gemm(filters, buffer, OP_NO_TRANS, OP_NO_TRANS, m, n, k, (T) 1.0, (T) 0.0,
         convout_img);
  }
}
template void conv_bc01_matmul<float>(const float *imgs, const float *filters,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, float *convout);



template <typename T>
void conv_bc01_matmul_bprop_imgs(const T *filters, const T *convout_d,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, T *imgs_d) {
  int convout_h = (img_h + 2 * pad_y - filter_h) / stride_y + 1;
  int convout_w = (img_w + 2 * pad_x - filter_w) / stride_x + 1;
  int win_size = filter_h * filter_w;
  T *buffer = (T *) CUDA::buffer(sizeof(float) * n_channels * win_size
                                 * convout_h * convout_w);
  int m = n_channels * win_size;
  int k = n_filters;
  int n = convout_h * convout_w;
  for (int i = 0; i < n_imgs; ++i) {
    const T *convout_img_d = convout_d + i * n_filters * convout_h * convout_w;
    gemm(filters, convout_img_d, OP_TRANS, OP_NO_TRANS, m, n, k, (T) 1.0,
         (T) 0.0, buffer);

    T *img_d = imgs_d + i * n_channels * img_h * img_w;
    win2img(buffer, n_channels, img_h, img_w, filter_h, filter_w, pad_y, pad_x,
            stride_y, stride_x, img_d);
  }
}
template void conv_bc01_matmul_bprop_imgs(const float *filters, const float *convout_d,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, float *imgs_d);



template <typename T>
void conv_bc01_matmul_bprop_filters(const T *imgs, const T *convout_d,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, T *filters_d) {
  int convout_h = (img_h + 2 * pad_y - filter_h) / stride_y + 1;
  int convout_w = (img_w + 2 * pad_x - filter_w) / stride_x + 1;
  int win_size = filter_h * filter_w;
  T *buffer = (T *) CUDA::buffer(sizeof(float) * n_channels * win_size
                                 * convout_h * convout_w);
  int m = n_filters;
  int k = convout_h * convout_w;
  int n = n_channels*win_size;
  for (int i = 0; i < n_imgs; ++i) {
    const T *img = imgs + i * n_channels * img_h * img_w;
    img2win(img, n_channels, img_h, img_w, filter_h, filter_w, pad_y, pad_x,
            stride_y, stride_x, buffer);
    const T *convout_img_d = convout_d + i * n_filters * convout_h * convout_w;
    T beta = i > 0 ? 1.0 : 0.0;
    gemm(convout_img_d, buffer, OP_NO_TRANS, OP_TRANS, m, n, k, (T) 1.0, beta,
         filters_d);
  }
}
template void conv_bc01_matmul_bprop_filters(const float *imgs, const float *convout_d,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, float *filters_d);

}
