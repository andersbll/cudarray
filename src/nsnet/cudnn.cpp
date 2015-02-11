#ifdef CUDNN_ENABLED

#include <iostream>
#include "cudarray/common.hpp"
#include "cudarray/nnet/cudnn.hpp"

namespace cudarray {


const float CUDNN::one = 1.0f;
const float CUDNN::zero = 0.0f;


template <typename T>
PoolBC01CuDNN<T>::PoolBC01CuDNN(int win_h, int win_w, int pad_y, int pad_x,
    int stride_y, int stride_x) : win_h(win_h), win_w(win_w), pad_y(pad_y),
    pad_x(pad_x), stride_y(stride_y), stride_x(stride_x), n_imgs(0),
    n_channels(0), img_h(0), img_w(0) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&imgs_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&poolout_desc));
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
}


template <typename T>
PoolBC01CuDNN<T>::~PoolBC01CuDNN() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(imgs_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(poolout_desc));
  CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
}


template <typename T>
void PoolBC01CuDNN<T>::fprop(const T *imgs, int n_imgs, int n_channels,
    int img_h, int img_w, T *poolout) {
  if (n_imgs != this->n_imgs || n_channels != this->n_channels ||
      img_h != this->img_h || img_w != this->img_w) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        imgs_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_imgs, n_channels,
        img_h, img_w
    ));

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc, CUDNN_POOLING_MAX, win_h, win_w, pad_y, pad_x, stride_y,
        stride_x
    ));
    int poolout_h = (img_h + 2*pad_y - win_h) / stride_y + 1;
    int poolout_w = (img_w + 2*pad_x - win_w) / stride_x + 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        poolout_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_imgs, n_channels,
        poolout_h, poolout_w
    ));
    this->n_imgs = n_imgs;
    this->n_channels = n_channels;
    this->img_h = img_h;
    this->img_w = img_w;
  }

  CUDNN_CHECK(cudnnPoolingForward(
      CUDNN::handle(), pool_desc, &CUDNN::one, imgs_desc, imgs, &CUDNN::zero,
      poolout_desc, poolout
  ));
}


template <typename T>
void PoolBC01CuDNN<T>::bprop(const T *imgs, const T* poolout,
                             const T *poolout_d, T *imgs_d) {
  CUDNN_CHECK(cudnnPoolingBackward(
    CUDNN::handle(), pool_desc, &CUDNN::one, poolout_desc, poolout, poolout_desc,
    poolout_d, imgs_desc, imgs, &CUDNN::zero, imgs_desc, imgs_d
  ));
}


template class PoolBC01CuDNN<float>;



template <typename T>
ConvBC01CuDNN<T>::ConvBC01CuDNN(int pad_y, int pad_x, int stride_y,
    int stride_x) : pad_y(pad_y), pad_x(pad_x), stride_y(stride_y),
    stride_x(stride_x), n_imgs(0), n_channels(0), n_filters(0), img_h(0),
    img_w(0), filter_h(0), filter_w(0), workspace_size(0), workspace(NULL) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&imgs_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&convout_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filters_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
}


template <typename T>
ConvBC01CuDNN<T>::~ConvBC01CuDNN() {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(imgs_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(convout_desc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filters_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}


template <typename T>
void ConvBC01CuDNN<T>::fprop(const T *imgs, const T *filters, int n_imgs,
    int n_channels, int n_filters, int img_h, int img_w, int filter_h,
    int filter_w, T *convout) {
  bool set_conv_desc = false;
  if (n_imgs != this->n_imgs || n_channels != this->n_channels ||
      img_h != this->img_h || img_w != this->img_w) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        imgs_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_imgs, n_channels,
        img_h, img_w
    ));
    this->n_imgs = n_imgs;
    this->n_channels = n_channels;
    this->img_h = img_h;
    this->img_w = img_w;
    set_conv_desc = true;
  }
  if (n_filters != this->n_filters || n_channels != this->n_channels ||
      filter_h != this->filter_h || filter_w != this->filter_w) {
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filters_desc, CUDNN_DATA_FLOAT, n_filters, n_channels, filter_h,
        filter_w
    ));
    this->n_filters = n_filters;
    this->n_channels = n_channels;
    this->filter_h = filter_h;
    this->filter_w = filter_w;
    set_conv_desc = true;
  }
  if (set_conv_desc) {
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, pad_y, pad_x, stride_y, stride_x, 1, 1, CUDNN_CONVOLUTION
    ));
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, imgs_desc, filters_desc, &n, &c, &h, &w
    ));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        convout_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w
    ));
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        CUDNN::handle(), imgs_desc, filters_desc, conv_desc, convout_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo
    ));
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        CUDNN::handle(), imgs_desc, filters_desc, conv_desc, convout_desc,
        fwd_algo, &workspace_size
    ));

    if (workspace_size > 0) {
      workspace = CUDA::buffer(workspace_size);
    } else {
      workspace = 0;
    }
  }
  CUDNN_CHECK(cudnnConvolutionForward(
      CUDNN::handle(), &CUDNN::one, imgs_desc, imgs, filters_desc, filters,
      conv_desc, fwd_algo, workspace, workspace_size, &CUDNN::zero,
      convout_desc, convout
  ));
}


template <typename T>
void ConvBC01CuDNN<T>::bprop(const T* imgs, const T* filters,
                             const T *convout_d, T *imgs_d, T *filters_d) {
  if (filters_d) {
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      CUDNN::handle(), &CUDNN::one, imgs_desc, imgs, convout_desc, convout_d,
      conv_desc, &CUDNN::zero, filters_desc, filters_d
    ));
  }
  if (imgs_d) {
    CUDNN_CHECK(cudnnConvolutionBackwardData(
      CUDNN::handle(), &CUDNN::one, filters_desc, filters, convout_desc,
      convout_d, conv_desc, &CUDNN::zero, imgs_desc, imgs_d
    ));
  }
}

template class ConvBC01CuDNN<float>;


const char *cudnn_message(cudnnStatus_t status){
  switch(status) {
    case CUDNN_STATUS_SUCCESS:
      return "The operation completed successfully";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "The cuDNN library was not initialized properly.";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "Resource allocation failed inside the cuDNN library.";
    case CUDNN_STATUS_BAD_PARAM:
      return "An incorrect parameter was passed to the function.";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "An internal cuDNN operation failed.";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "The function requires a feature absent from the GPU";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "An access to GPU memory space failed.";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute.";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "The functionality not presently supported by cuDNN.";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "The functionality requested requires some license.";
    default:
      throw std::runtime_error("invalid cudnnStatus_t");
  }
}


} // cudarray

#endif // CUDNN_ENABLED
