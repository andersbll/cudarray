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
              np.ndarray[DTYPE_t, ndim=4] convout = None):
    """ Multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """
    # TODO: support padding and striding  

    cdef uint n_imgs = imgs.shape[0]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]
    cdef uint n_channels_in = filters.shape[0]
    cdef uint n_channels_out = filters.shape[1]
    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]

    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    if n_channels_in != imgs.shape[1]:
        raise ValueError('Mismatch in number of channels between filters and imgs.')

    if convout == None:
        # ? cdef np.ndarray[DTYPE_t, ndim=4] 
        convout = np.ndarray(shape=(n_imgs, n_channels_out, img_h, img_w), dtype=DTYPE)
    else:
        if n_channels_out != convout.shape[1]:
            raise ValueError('Mismatch in number of channels between filters and convout.')
        if n_imgs != convout.shape[0]:
            raise ValueError('Mismatch in number of images between imgs and convout.')
        if img_h != convout.shape[2] or img_w != convout.shape[3]:
            raise ValueError('Mismatch in image shape between imgs and convout.')


    #if fil_h % 2 != 1 or fil_w % 2 != 1:
        #raise ValueError('Only odd filter dimensions are supported.')
    


    cdef uint i, c_in, c_out
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t value

    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off, mid_off_h, mid_off_w

#mid_off only add one to max iff filter is of an uneaven sice 
    mid_off_h = 0
    mid_off_w = 0
    if mid_off_h % 2 == 1:
        mid_off_h = 1 

    if mid_off_w % 2 == 1:
        mid_off_w = 1 

#    with nogil, parallel(num_threads=8):
#        for i in prange(n_imgs):
#            value = 0.0
    for i in range(n_imgs):
        for c_out in range(n_channels_out):
            for y in range(img_h):
                y_off_min = int_max(-y, -fil_mid_h)
                y_off_max = int_min(img_h-y, fil_mid_h+mid_off_h)
                for x in range(img_w):
                    x_off_min = int_max(-x, -fil_mid_w)
                    x_off_max = int_min(img_w-x, fil_mid_w+mid_off_w)
                    value = 0.0
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = <uint>(y + y_off)
                            img_x = <uint>(x + x_off)
                            fil_y = <uint>(fil_mid_w + y_off)
                            fil_x = <uint>(fil_mid_h + x_off)
                            for c_in in range(n_channels_in):
                                value += imgs[i, c_in, img_y, img_x] * filters[c_in, c_out, fil_y, fil_x]
                    convout[i, c_out, y, x] = value

    return convout


@cython.boundscheck(False)
@cython.wraparound(False)
def conv_bc01_bprop_filters(np.ndarray[DTYPE_t, ndim=4] imgs,
                    np.ndarray[DTYPE_t, ndim=4] convout_d,
                    np.ndarray[DTYPE_t, ndim=4] filters_d = None):
    """ Back-propagate gradients of multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """
    cdef uint n_imgs = imgs.shape[0]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]
    cdef uint n_channels_convout = convout_d.shape[1]
    cdef uint n_channels_imgs = imgs.shape[1]
    cdef uint fil_h = convout_d.shape[2]
    cdef uint fil_w = convout_d.shape[3]
    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint i, c_convout, c_imgs
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t convout_d_value
    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off, mid_off_h, mid_off_w

#mid_off only add one to max iff filter is of an uneaven sice 
    mid_off_h = 0
    mid_off_w = 0
    if mid_off_h % 2 == 1:
        mid_off_h = 1 

    if mid_off_w % 2 == 1:
        mid_off_w = 1 

    if filters_d == None:
        filters_d = np.zeros(shape=(n_channels_imgs, n_channels_imgs, img_h, img_w), dtype=DTYPE)
    else:
        filters_d[...] = 0

    for i in range(n_imgs):
        for c_convout in range(n_channels_convout):
            for y in range(img_h):
                y_off_min = int_max(-y, -fil_mid_h)
                y_off_max = int_min(img_h-y, fil_mid_h+mid_off_h)
                for x in range(img_w):
                    convout_d_value = convout_d[i, c_convout, y, x]
                    x_off_min = int_max(-x, -fil_mid_w)
                    x_off_max = int_min(img_w-x, fil_mid_w+mid_off_w)
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = <uint>(y + y_off)
                            img_x = <uint>(x + x_off)
                            fil_y = <uint>(fil_mid_w + y_off)
                            fil_x = <uint>(fil_mid_h + x_off)
                            for c_imgs in range(n_channels_imgs):
                                filters_d[c_imgs, c_convout, fil_y, fil_x] += imgs[i, c_imgs, img_y, img_x] * convout_d_value
#    filters_grad[...] /= n_imgs

@cython.boundscheck(False)
@cython.wraparound(False)
def conv_bc01_bprop_imgs(np.ndarray[DTYPE_t, ndim=4] filters,
                    np.ndarray[DTYPE_t, ndim=4] convout_d,
                    np.ndarray[DTYPE_t, ndim=4] imgs_d = None):
    """ Back-propagate gradients of multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """
    cdef uint n_imgs = convout_d.shape[0]
    cdef uint img_h = convout_d.shape[2]
    cdef uint img_w = convout_d.shape[3]
    cdef uint n_channels_convout = filters.shape[1]
    cdef uint n_channels_imgs = filters.shape[0]
    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]
    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint i, c_convout, c_imgs
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t convout_d_value
    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off, mid_off_h, mid_off_w

#mid_off only add one to max iff filter is of an uneaven sice 
    mid_off_h = 0
    mid_off_w = 0
    if mid_off_h % 2 == 1:
        mid_off_h = 1 

    if mid_off_w % 2 == 1:
        mid_off_w = 1 

    if imgs_d == None:
        imgs_d = np.zeros(shape=(n_imgs, n_channels_imgs, img_h, img_w), dtype=DTYPE)
    else:
        imgs_d[...] = 0

    for i in range(n_imgs):
        for c_convout in range(n_channels_convout):
            for y in range(img_h):
                y_off_min = int_max(-y, -fil_mid_h)
                y_off_max = int_min(img_h-y, fil_mid_h+mid_off_h)
                for x in range(img_w):
                    convout_d_value = convout_d[i, c_convout, y, x]
                    x_off_min = int_max(-x, -fil_mid_w)
                    x_off_max = int_min(img_w-x, fil_mid_w+mid_off_w)
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = <uint>(y + y_off)
                            img_x = <uint>(x + x_off)
                            fil_y = <uint>(fil_mid_w + y_off)
                            fil_x = <uint>(fil_mid_h + x_off)
                            for c_imgs in range(n_channels_imgs):
                                imgs_d[i, c_imgs, img_y, img_x] += filters[c_imgs, c_convout, fil_y, fil_x] * convout_d_value
    return imgs_d
#    filters_grad[...] /= n_imgs


@cython.boundscheck(False)
@cython.wraparound(False)
def conv_bc01_bprop(np.ndarray[DTYPE_t, ndim=4] imgs,
                    np.ndarray[DTYPE_t, ndim=4] convout_d,
                    np.ndarray[DTYPE_t, ndim=4] filters,
                    np.ndarray[DTYPE_t, ndim=4] imgs_grad,
                    np.ndarray[DTYPE_t, ndim=4] filters_grad):
    """ Back-propagate gradients of multi-image, multi-channel convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """

    cdef uint n_imgs = convout_d.shape[0]
    cdef uint img_h = convout_d.shape[2]
    cdef uint img_w = convout_d.shape[3]
    cdef uint n_channels_convout = filters.shape[1]
    cdef uint n_channels_imgs = filters.shape[0]
    cdef uint fil_h = filters.shape[2]
    cdef uint fil_w = filters.shape[3]
    cdef int fil_mid_h = fil_h // 2
    cdef int fil_mid_w = fil_w // 2

    cdef uint i, c_convout, c_imgs
    cdef uint img_y, img_x, fil_y, fil_x
    cdef DTYPE_t convout_d_value
    cdef int y, x, y_off_min, y_off_max, y_off, x_off_min, x_off_max, x_off, mid_off_h, mid_off_w

#mid_off only add one to max iff filter is of an uneaven sice 
    mid_off_h = 0
    mid_off_w = 0
    if mid_off_h % 2 == 1:
        mid_off_h = 1 

    if mid_off_w % 2 == 1:
        mid_off_w = 1 

    imgs_grad[...] = 0
    filters_grad[...] = 0
    for i in range(n_imgs):
        for c_convout in range(n_channels_convout):
            for y in range(img_h):
                y_off_min = int_max(-y, -fil_mid_h)
                y_off_max = int_min(img_h-y, fil_mid_h+mid_off_h)
                for x in range(img_w):
                    convout_d_value = convout_d[i, c_convout, y, x]
                    x_off_min = int_max(-x, -fil_mid_w)
                    x_off_max = int_min(img_w-x, fil_mid_w+mid_off_w)
                    for y_off in range(y_off_min, y_off_max):
                        for x_off in range(x_off_min, x_off_max):
                            img_y = <uint>(y + y_off)
                            img_x = <uint>(x + x_off)
                            fil_y = <uint>(fil_mid_w + y_off)
                            fil_x = <uint>(fil_mid_h + x_off)
                            for c_imgs in range(n_channels_imgs):
                                imgs_grad[i, c_imgs, img_y, img_x] += filters[c_imgs, c_convout, fil_y, fil_x] * convout_d_value
                                filters_grad[c_imgs, c_convout, fil_y, fil_x] += imgs[i, c_imgs, img_y, img_x] * convout_d_value
#    filters_grad[...] /= n_imgs
