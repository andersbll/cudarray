from __future__ import division
import numpy as np
import cython
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint


cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
def conv_seg_bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
              np.ndarray[DTYPE_t, ndim=4] filters,
              np.ndarray[DTYPE_t, ndim=4] convout):
    """ Multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_fragments , n_channels_in, img_h, img_w)
    filters has shape (n_channels_out, n_channels_in, filter_h, filter_w)
    """
    cdef uint F = imgs.shape[0]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]
    cdef uint n_channels_in = filters.shape[1]
    cdef uint n_channels_out = filters.shape[0]
    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]

    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint c_in, c_out, fg
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t value

    cdef int y, x, yMin, yMax, xMin, xMax
    
    """mid_off only add one to max iff filter is of an uneaven sice 
    This is done because filters of uneaven size have center shifte one Back-propagate
    [ 1, 1 , x , 1] wher x is center for a 1X4 filter"""
    mid_off_h = fil_h % 2
    mid_off_w = fil_w % 2

    for fg in range(F):
        for c_out in range(n_channels_out):
            for y in range(img_h):
                for x in range(img_w):
                    value = 0.0
                    fil_y = 0

                    yMin = y-fil_mid_h
                    yMax = y+fil_mid_h+mid_off_h
                    for y_set in range(yMin, yMax): 
                        img_y = getImgIndex(y_set, img_h)
                        fil_x = 0

                        xMin = x-fil_mid_w
                        xMax = x+fil_mid_w+mid_off_w
                        for x_set in range(xMin, xMax):   
                            img_x = getImgIndex(x_set, img_w)

                            for c_in in range(n_channels_in):
                                value += imgs[fg, c_in, img_y, img_x] * filters[c_out, c_in, fil_y, fil_x]
                            fil_x += 1
                        fil_y += 1
                    convout[fg, c_out, y, x] = value
    return convout

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline getImgIndex(int tempIndex, uint size):
    cdef uint index
    if(tempIndex < 0):        
        index = <uint>(tempIndex * -1)
    elif(tempIndex >= size):
        index = <uint>(size - (tempIndex % size) - 1)
    else:
        index = <uint>(tempIndex)
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
def conv_seg_bc01_bprop(np.ndarray[DTYPE_t, ndim=4] imgs,
                    np.ndarray[DTYPE_t, ndim=4] convout_d,
                    np.ndarray[DTYPE_t, ndim=4] filters,
                    np.ndarray[DTYPE_t, ndim=4] imgs_grad,
                    np.ndarray[DTYPE_t, ndim=4] filters_grad):
    """ Back-propagate gradients of multi-image, multi-channel convolution
    imgs has shape (b, fg, c, img_h, img_w)
    filters has shape (f, c_filters, img_h, img_w)
    convout has shape (b_convout, f_convout, img_h, img_w)
    """
    cdef uint img_channels = imgs.shape[1]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]

    cdef uint f_convout = convout_d.shape[1]
    cdef uint convout_d_h = convout_d.shape[2]
    cdef uint convout_d_w = convout_d.shape[3]

    cdef uint F_out = convout_d.shape[0]

    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]
    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint c_convout, c_imgs
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t convout_d_value
    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max 
    cdef int x_off, mid_off_h, mid_off_w, img_x_center, img_y_center
    
    """mid_off only add one to max iff filter is of an uneaven sice 
    This is done because filters of uneaven size have center shifte one Back-propagate
    [ 1, 1 , x , 1] wher x is center for a 1X4 filter"""
    mid_off_h = fil_h % 2
    mid_off_w = fil_w % 2

    imgs_grad[...] = 0
    filters_grad[...] = 0

    for fg in range(F_out):
        for c_out in range(f_convout):
            for y in range(img_h):
                for x in range(img_w):
                    convout_d_value = convout_d[fg, c_convout, y, x]
                    fil_y = 0
                    yMin = y-fil_mid_h
                    yMax = y+fil_mid_h+mid_off_h
                    for y_set in range(yMin, yMax): 
                        img_y = getImgIndex(y_set, img_h)
                        fil_x = 0
                        xMin = x-fil_mid_w
                        xMax = x+fil_mid_w+mid_off_w
                        for x_set in range(xMin, xMax):   
                            img_x = getImgIndex(x_set, img_w)
                            for c_imgs in range(img_channels):
                                imgs_grad[fg, c_imgs, img_y, img_x] += filters[c_out, c_imgs, fil_y, fil_x] * convout_d_value
                                filters_grad[c_out, c_imgs, fil_y, fil_x] += imgs[fg, c_imgs, img_y, img_x] * convout_d_value

                            fil_x += 1
                        fil_y += 1
