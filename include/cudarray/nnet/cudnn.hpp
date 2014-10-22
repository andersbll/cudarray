#ifndef CUDNN_HPP_
#define CUDNN_HPP_

#include <sstream>
#include <stdexcept>
#include <cudnn.h>

namespace cudarray {


template <typename T>
class PoolBC01CuDNN {
public:
  PoolBC01CuDNN(int win_h, int win_w, int pad_y, int pad_x, int stride_y,
                int stride_x);
  ~PoolBC01CuDNN();

  void fprop(const T *imgs, int n_imgs, int n_channels, int img_h, int img_w,
             T *poolout);

  void bprop(const T *imgs, const T* poolout, const T *poolout_d, T *imgs_d);

private:
  int win_h;
  int win_w;
  int pad_y;
  int pad_x;
  int stride_y;
  int stride_x;
  int n_imgs;
  int n_channels;
  int img_h;
  int img_w;
  cudnnTensor4dDescriptor_t imgs_desc;
  cudnnTensor4dDescriptor_t poolout_desc;
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
  cudnnTensor4dDescriptor_t imgs_desc;
  cudnnTensor4dDescriptor_t convout_desc;
  cudnnFilterDescriptor_t filters_desc;
  cudnnConvolutionDescriptor_t conv_desc;
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

}

#endif // CUDNN_HPP_
