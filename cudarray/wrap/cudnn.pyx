from cpython cimport array as c_array
from array import array
cimport numpy as np
from .array_data cimport ArrayData, float_ptr
cimport cudnn



cdef class PoolBC01CuDNN_f:
    cdef PoolBC01CuDNN[float] *ptr
    cdef tuple win_shape
    cdef tuple padding
    cdef tuple strides
    cdef str mode

    def __init__(self, win_shape, padding, strides, mode):
        cdef c_array.array win_shape_ = array('i', win_shape)
        cdef c_array.array padding_ = array('i', padding)
        cdef c_array.array strides_ = array('i', strides)
        self.win_shape = win_shape
        self.padding = padding
        self.strides = strides
        self.mode = mode
        if mode == 'avg':
            mode = POOL_AVG
        elif mode == 'max':
            mode = POOL_MAX
        else:
            raise ValueError('Invalid mode: %s' % mode)
        self.ptr = new PoolBC01CuDNN[float](
            len(win_shape), win_shape_.data.as_ints, padding_.data.as_ints,
            strides_.data.as_ints, mode
        )

    def __dealloc__(self):
        del self.ptr

    def __reduce__(self):
        args = (self.win_shape, self.padding, self.strides, self.mode)
        return (PoolBC01CuDNN_f, args)

    def fprop(self, ArrayData imgs, imgs_shape, ArrayData poolout):
        cdef c_array.array imgs_shape_ = array('i', imgs_shape)
        self.ptr.fprop(float_ptr(imgs), imgs_shape_.data.as_ints,
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
    cdef tuple padding
    cdef tuple strides
    def __init__(self, padding, strides):
        self.padding = padding
        self.strides = strides
        self.ptr = new ConvBC01CuDNN[float](
            padding[0], padding[1], strides[0], strides[1]
        )

    def __dealloc__(self):
        del self.ptr

    def __reduce__(self):
        args = (self.padding, self.strides)
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
