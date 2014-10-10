cimport numpy as np

cimport nnet
from .array_data cimport ArrayData


def _conv_bc01_matmul(ArrayData imgs, ArrayData filters, int n_imgs,
    int n_channels, int n_filters, img_shape, filter_shape, padding, strides,
    ArrayData convout):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int filter_h = filter_shape[0]
    cdef int filter_w = filter_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]

    if imgs.dtype == np.dtype('float32'):
        nnet.conv_bc01_matmul(<float *>imgs.dev_ptr, <float *>filters.dev_ptr,
            n_imgs, n_channels, n_filters, img_h, img_w, filter_h, filter_w,
            pad_y, pad_x, stride_y, stride_x, <float *>convout.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(imgs.dtype))


def _conv_bc01_matmul_bprop_filters(ArrayData imgs, ArrayData convout_d,
    int n_imgs, int n_channels, int n_filters, img_shape, filter_shape,
    padding, strides, ArrayData filters_d):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int filter_h = filter_shape[0]
    cdef int filter_w = filter_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]
    if imgs.dtype == np.dtype('float32'):
        nnet.conv_bc01_matmul_bprop_filters(<float *>imgs.dev_ptr, <float *>convout_d.dev_ptr,
            n_imgs, n_channels, n_filters, img_h, img_w, filter_h, filter_w,
            pad_y, pad_x, stride_y, stride_x, <float *>filters_d.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(imgs.dtype))


def _conv_bc01_matmul_bprop_imgs(ArrayData filters, ArrayData convout_d,
    int n_imgs, int n_channels, int n_filters, img_shape, filter_shape,
    padding, strides, ArrayData imgs_d):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int filter_h = filter_shape[0]
    cdef int filter_w = filter_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]
    if filters.dtype == np.dtype('float32'):
        nnet.conv_bc01_matmul_bprop_imgs(<float *>filters.dev_ptr, <float *>convout_d.dev_ptr,
            n_imgs, n_channels, n_filters, img_h, img_w, filter_h, filter_w,
            pad_y, pad_x, stride_y, stride_x, <float *>imgs_d.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(filters.dtype))


def _max_pool_b01(ArrayData imgs, int n_imgs, img_shape, win_shape, padding,
    strides, ArrayData out, ArrayData mask):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int win_h = win_shape[0]
    cdef int win_w = win_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]
    if imgs.dtype == np.dtype('float32'):
        nnet.max_pool_b01(<float *>imgs.dev_ptr, n_imgs, img_h, img_w,
            win_h, win_w, pad_y, pad_x, stride_y, stride_x,
            <float *>out.dev_ptr, <int *>mask.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(imgs.dtype))


def _max_pool_b01_bprop(ArrayData out_d, ArrayData mask, int n_imgs, img_shape,
    win_shape, padding, strides, ArrayData imgs_d):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int win_h = win_shape[0]
    cdef int win_w = win_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]
    if out_d.dtype == np.dtype('float32'):
        nnet.max_pool_b01_bprob(<float *>out_d.dev_ptr, <int *>mask.dev_ptr,
            n_imgs, img_h, img_w, win_h, win_w, pad_y, pad_x, stride_y,
            stride_x, <float *> imgs_d.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(out_d.dtype))

