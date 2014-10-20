from __future__ import division
import numpy as np
import cython
#from cython.parallel import parallel, prange, threadlocal
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint


cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
def conv_bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
              np.ndarray[DTYPE_t, ndim=4] filters,
              tuple padding,
              tuple strides,
              np.ndarray[DTYPE_t, ndim=4] convout):
    """ Multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_out, n_channels_in, filter_h, filter_w)
    """
    # TODO: support padding and striding  

    cdef uint n_imgs = imgs.shape[0]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]
    cdef uint n_channels_in = filters.shape[1]
    cdef uint n_channels_out = filters.shape[0]
    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]

    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint i, c_in, c_out
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t value

    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off, mid_off_h, mid_off_w, img_x_center, img_y_center
    
    """mid_off only add one to max iff filter is of an uneaven sice 
    This is done because filters of uneaven size have center shifte one Back-propagate
    [ 1, 1 , x , 1] wher x is center for a 1X4 filter"""
    mid_off_h = fil_h % 2
    mid_off_w = fil_w % 2

    cdef uint stride_h = strides[0]
    cdef uint stride_w = strides[1]

    cdef uint padding_h = padding[0]
    cdef uint padding_w = padding[1]

    cdef uint out_h = convout.shape[2]
    cdef uint out_w = convout.shape[3]

    for i in range(n_imgs):
        for c_out in range(n_channels_out):
            for y in range(out_h):
                img_y_center = y*stride_h+fil_mid_h
                y_off_min = int_max(-img_y_center, -padding_h-fil_mid_h)
                y_off_max = int_min(img_h-img_y_center, fil_mid_h+mid_off_h-padding_h)
                for x in range(out_w):
                    img_x_center = x*stride_w+fil_mid_w
                    x_off_min = int_max(-img_x_center, -padding_w-fil_mid_w)
                    x_off_max = int_min(img_w-img_x_center, fil_mid_w+mid_off_w-padding_w)
                    value = 0.0
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = <uint>(img_y_center + y_off)
                            img_x = <uint>(img_x_center + x_off)
                            fil_y = <uint>(fil_mid_h + padding_h + y_off)
                            fil_x = <uint>(fil_mid_w + padding_w + x_off)
                            for c_in in range(n_channels_in):
                                value += imgs[i, c_in, img_y, img_x] * filters[c_out, c_in, fil_y, fil_x]
                    convout[i, c_out, y, x] = value

    return convout

@cython.boundscheck(False)
@cython.wraparound(False)
def conv_bc01_bprop(np.ndarray[DTYPE_t, ndim=4] imgs,
                    np.ndarray[DTYPE_t, ndim=4] convout_d,
                    np.ndarray[DTYPE_t, ndim=4] filters,
                    tuple padding,
                    tuple strides,
                    np.ndarray[DTYPE_t, ndim=4] imgs_grad,
                    np.ndarray[DTYPE_t, ndim=4] filters_grad):
    """ Back-propagate gradients of multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_out, n_channels_in, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """

    cdef uint n_imgs = convout_d.shape[0]
    cdef uint img_h = convout_d.shape[2]
    cdef uint img_w = convout_d.shape[3]
    cdef uint n_channels_convout = filters.shape[0]
    cdef uint n_channels_imgs = filters.shape[1]
    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]
    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint i, c_convout, c_imgs
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t convout_d_value
    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max 
    cdef int x_off, mid_off_h, mid_off_w, img_x_center, img_y_center
    
    """mid_off only add one to max iff filter is of an uneaven sice 
    This is done because filters of uneaven size have center shifte one Back-propagate
    [ 1, 1 , x , 1] wher x is center for a 1X4 filter"""
    mid_off_h = fil_h % 2
    mid_off_w = fil_w % 2

    cdef uint stride_h = strides[0]
    cdef uint stride_w = strides[1]

    cdef uint padding_h = padding[0]
    cdef uint padding_w = padding[1]

    imgs_grad[...] = 0
    filters_grad[...] = 0
    for i in range(n_imgs):
        for c_convout in range(n_channels_convout):
            for y in range(img_h):
                img_y_center = y*stride_h+fil_mid_h
                y_off_min = int_max(-img_y_center, -padding_h-fil_mid_h)
                y_off_max = int_min(img_h-img_y_center, fil_mid_h+mid_off_h-padding_h)
                for x in range(img_w):
                    convout_d_value = convout_d[i, c_convout, y, x]
                    img_x_center = x*stride_w+fil_mid_w
                    x_off_min = int_max(-img_x_center, -padding_w-fil_mid_w)
                    x_off_max = int_min(img_w-img_x_center, fil_mid_w+mid_off_w-padding_w)
                    value = 0.0
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = <uint>(img_y_center + y_off)
                            img_x = <uint>(img_x_center + x_off)
                            fil_y = <uint>(fil_mid_h + padding_h + y_off)
                            fil_x = <uint>(fil_mid_w + padding_w + x_off)
                            for c_imgs in range(n_channels_imgs):
                                imgs_grad[i, c_imgs, img_y, img_x] += filters[c_imgs, c_convout, fil_y, fil_x] * convout_d_value
                                filters_grad[c_convout, c_imgs, fil_y, fil_x] += imgs[i, c_imgs, img_y, img_x] * convout_d_value
#    filters_grad[...] /= n_imgs
