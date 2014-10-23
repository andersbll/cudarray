cimport numpy as np
from .array_data cimport ArrayData
cimport cudnn


cdef float *float_ptr(array):
    cdef float *ptr;
    if array is None:
        ptr = NULL
    else:
        ptr = <float *> (<ArrayData> array._data).dev_ptr
    return ptr


cdef class PoolBC01CuDNN_f:
    cdef PoolBC01CuDNN[float] *ptr
    def __init__(self, win_shape, padding, strides):
        cdef int win_w = win_shape[0]
        cdef int win_h = win_shape[1]
        cdef int pad_y = padding[0]
        cdef int pad_x = padding[1]
        cdef int stride_y = strides[0]
        cdef int stride_x = strides[1]
        self.ptr = new PoolBC01CuDNN[float](
            win_w, win_h, pad_y, pad_x, stride_y, stride_x
        )

    def __dealloc__(self):
        del self.ptr

    def fprop(self, ArrayData imgs, int n_imgs, int n_channels, img_shape,
              ArrayData poolout):
        cdef int img_h = img_shape[0]
        cdef int img_w = img_shape[1]
        self.ptr.fprop(
            <const float *> imgs.dev_ptr, n_imgs, n_channels, img_h, img_w,
            <float *> poolout.dev_ptr
        )

    def bprop(self, ArrayData imgs, ArrayData poolout, ArrayData poolout_d,
              ArrayData imgs_d):
        self.ptr.bprop(
            <const float *> imgs.dev_ptr,
            <const float *> poolout.dev_ptr,
            <const float *> poolout_d.dev_ptr, <float *> imgs_d.dev_ptr
        )


cdef class ConvBC01CuDNN_f:
    cdef ConvBC01CuDNN[float] *ptr
    def __init__(self, padding, strides):
        cdef int pad_y = padding[0]
        cdef int pad_x = padding[1]
        cdef int stride_y = strides[0]
        cdef int stride_x = strides[1]
        self.ptr = new ConvBC01CuDNN[float](
            pad_y, pad_x, stride_y, stride_x
        )

    def __dealloc__(self):
        del self.ptr

    def fprop(self, ArrayData imgs, ArrayData filters, int n_imgs,
              int n_channels, int n_filters, img_shape, filter_shape,
              ArrayData convout):
        cdef int img_h = img_shape[0]
        cdef int img_w = img_shape[1]
        cdef int filter_h = filter_shape[0]
        cdef int filter_w = filter_shape[1]
        self.ptr.fprop(
            <const float *> imgs.dev_ptr, <const float *> filters.dev_ptr,
            n_imgs, n_channels, n_filters, img_h, img_w, filter_h, filter_w,
            <float *> convout.dev_ptr
        )

    def bprop(self, imgs, filters, convout_d, imgs_d, filters_d):
        self.ptr.bprop(float_ptr(imgs), float_ptr(filters),
            float_ptr(convout_d), float_ptr(imgs_d), float_ptr(filters_d))


def conv_bc01_cudnn(padding, strides):
    # TODO: only float is supported
    return ConvBC01CuDNN_f(padding, strides)
