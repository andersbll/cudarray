cimport numpy as np
from .array_data cimport ArrayData, float_ptr
cimport cudnn


cdef class PoolBC01CuDNN_f:
    cdef PoolBC01CuDNN[float] *ptr
    cdef int win_w
    cdef int win_h
    cdef int pad_y
    cdef int pad_x
    cdef int stride_y
    cdef int stride_x

    def __init__(self, win_shape, padding, strides):
        self.win_w = win_shape[0]
        self.win_h = win_shape[1]
        self.pad_y = padding[0]
        self.pad_x = padding[1]
        self.stride_y = strides[0]
        self.stride_x = strides[1]
        self.ptr = new PoolBC01CuDNN[float](
            self.win_w, self.win_h, self.pad_y, self.pad_x, self.stride_y,
            self.stride_x
        )

    def __dealloc__(self):
        del self.ptr

    def __reduce__(self):
        args = ((self.win_w, self.win_h), (self.pad_y, self.pad_x),
                (self.stride_y, self.stride_x))
        return (PoolBC01CuDNN_f, args)

    def fprop(self, ArrayData imgs, int n_imgs, int n_channels, img_shape,
              ArrayData poolout):
        cdef int img_h = img_shape[0]
        cdef int img_w = img_shape[1]
        self.ptr.fprop(float_ptr(imgs), n_imgs, n_channels, img_h, img_w,
                       float_ptr(poolout))

    def bprop(self, ArrayData imgs, ArrayData poolout, ArrayData poolout_d,
              ArrayData imgs_d):
        self.ptr.bprop(
            <const float *> imgs.dev_ptr,
            <const float *> poolout.dev_ptr,
            <const float *> poolout_d.dev_ptr, <float *> imgs_d.dev_ptr
        )


cdef class ConvBC01CuDNN_f:
    cdef ConvBC01CuDNN[float] *ptr
    cdef int pad_y
    cdef int pad_x
    cdef int stride_y
    cdef int stride_x
    def __init__(self, padding, strides):
        self.pad_y = padding[0]
        self.pad_x = padding[1]
        self.stride_y = strides[0]
        self.stride_x = strides[1]
        self.ptr = new ConvBC01CuDNN[float](
            self.pad_y, self.pad_x, self.stride_y, self.stride_x
        )

    def __dealloc__(self):
        del self.ptr

    def __reduce__(self):
        args = ((self.pad_y, self.pad_x), (self.stride_y, self.stride_x))
        return (ConvBC01CuDNN_f, args)

    def fprop(self, ArrayData imgs, ArrayData filters, int n_imgs,
              int n_channels, int n_filters, img_shape, filter_shape,
              ArrayData convout):
        cdef int img_h = img_shape[0]
        cdef int img_w = img_shape[1]
        cdef int filter_h = filter_shape[0]
        cdef int filter_w = filter_shape[1]
        self.ptr.fprop(float_ptr(imgs), float_ptr(filters), n_imgs, n_channels,
            n_filters, img_h, img_w, filter_h, filter_w, float_ptr(convout))

    def bprop(self, ArrayData imgs, ArrayData filters, ArrayData convout_d,
              ArrayData imgs_d, ArrayData filters_d):
        cdef float *imgs_d_ptr = <float *>NULL if imgs_d is None \
                                               else float_ptr(imgs_d)
        cdef float *filters_d_ptr = <float *>NULL if filters_d is None \
                                                  else float_ptr(filters_d)
        self.ptr.bprop(float_ptr(imgs), float_ptr(filters),
                       float_ptr(convout_d), imgs_d_ptr, filters_d_ptr)


def conv_bc01_cudnn(padding, strides):
    # TODO: only float is supported
    return ConvBC01CuDNN_f(padding, strides)
