#ifdef CUDNN_ENABLED

#ifndef CUDNN_HPP_
#define CUDNN_HPP_

#include <sstream>
#include <stdexcept>
#include <cudnn.h>

namespace cudarray {

enum PoolMode {POOL_AVG, POOL_MAX};

const int MAX_IMG_DIMS = 3;
const int WORKSPACE_LIMIT = 1024*1024*1024;

template <typename T>
class PoolBC01CuDNN {
public:
  PoolBC01CuDNN(int n_img_dims, int *win_shape, int *padding, int *strides,
                PoolMode pool_mode);
  ~PoolBC01CuDNN();

  void fprop(const T *imgs, int *imgs_shape, T *poolout);

  void bprop(const T *imgs, const T* poolout, const T *poolout_d, T *imgs_d);

private:
  int n_img_dims;
  int win_shape[MAX_IMG_DIMS];
  int padding[MAX_IMG_DIMS];
  int strides[MAX_IMG_DIMS];
  int imgs_shape[MAX_IMG_DIMS + 2];
  cudnnPoolingMode_t pool_mode;
  cudnnTensorDescriptor_t imgs_desc;
  cudnnTensorDescriptor_t poolout_desc;
  cudnnPoolingDescriptor_t pool_desc;
};


template <typename T>
class ConvBC01CuDNN {
public:
  ConvBC01CuDNN(int pad_y, int pad_x, int stride_y, int stride_x);
  ~ConvBC01CuDNN();

  void fprop(const T *imgs, const T *filters, int n_imgs, int n_channels,
      int n_filters, int img_h, int img_w, int filter_h, int filter_w,
      T *convout);

  void bprop(const T* imgs, const T* filters, const T *convout_d, T *imgs_d,
             T *filters_d);

private:
  int pad_y;
  int pad_x;
  int stride_y;
  int stride_x;
  int n_imgs;
  int n_channels;
  int n_filters;
  int img_h;
  int img_w;
  int filter_h;
  int filter_w;
  cudnnTensorDescriptor_t imgs_desc;
  cudnnTensorDescriptor_t convout_desc;
  cudnnFilterDescriptor_t filters_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filters_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_imgs_algo;
  size_t workspace_size;
};


const char* cudnn_message(cudnnStatus_t status);

inline void cudnn_check(cudnnStatus_t status, const char *file, int line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    std::ostringstream o;
    o << file << ":" << line << ": " << cudnn_message(status);
    throw std::runtime_error(o.str());
  }
}

#define CUDNN_CHECK(status) { cudnn_check((status), __FILE__, __LINE__); }

/*
  Singleton class to handle cuDNN resources.
*/
class CUDNN {
public:
  static const float one;
  static const float zero;

  inline static CUDNN &instance() {
    static CUDNN instance_;
    return instance_;
  }

  inline static cudnnHandle_t &handle() {
    return instance().handle_;
  }

private:
  cudnnHandle_t handle_;
  CUDNN() {
    CUDNN_CHECK(cudnnCreate(&handle_));
  }
  ~CUDNN() {
    CUDNN_CHECK(cudnnDestroy(handle_));
  }
  CUDNN(CUDNN const&);
  void operator=(CUDNN const&);
};


} // cudarray

#endif // CUDNN_HPP_

#endif // CUDNN_ENABLED
