cimport numpy as np
cimport nnet
from .array_data cimport ArrayData, float_ptr, int_ptr, is_float


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
    if is_float(imgs):
        nnet.conv_bc01_matmul(float_ptr(imgs), float_ptr(filters),
            n_imgs, n_channels, n_filters, img_h, img_w, filter_h, filter_w,
            pad_y, pad_x, stride_y, stride_x, float_ptr(convout))
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
    if is_float(imgs):
        nnet.conv_bc01_matmul_bprop_filters(float_ptr(imgs),
            float_ptr(convout_d), n_imgs, n_channels, n_filters, img_h, img_w,
            filter_h, filter_w, pad_y, pad_x, stride_y, stride_x,
            float_ptr(filters_d))
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
    if is_float(filters):
        nnet.conv_bc01_matmul_bprop_imgs(float_ptr(filters),
            float_ptr(convout_d), n_imgs, n_channels, n_filters, img_h, img_w,
            filter_h, filter_w, pad_y, pad_x, stride_y, stride_x,
            float_ptr(imgs_d))
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
    if is_float(imgs):
        nnet.max_pool_b01(float_ptr(imgs), n_imgs, img_h, img_w,
            win_h, win_w, pad_y, pad_x, stride_y, stride_x,
            float_ptr(out), int_ptr(mask))
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
    if is_float(out_d):
        nnet.max_pool_b01_bprob(float_ptr(out_d), int_ptr(mask),
            n_imgs, img_h, img_w, win_h, win_w, pad_y, pad_x, stride_y,
            stride_x, float_ptr(imgs_d))
    else:
        raise ValueError('type %s not implemented' % str(out_d.dtype))


def _one_hot_encode(ArrayData labels, int n_classes, int n, ArrayData out):
    if is_float(out):
        nnet.one_hot_encode(int_ptr(labels), n_classes, n, float_ptr(out))
    else:
        raise ValueError('type %s not implemented' % str(out.dtype))
