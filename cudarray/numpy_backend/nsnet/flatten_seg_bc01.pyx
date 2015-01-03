from __future__ import division
import numpy as np
import cython
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint

cdef inline DTYPE_t dtype_t_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
def flatten_seg_bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
                         tuple win_shape,
                         np.ndarray[long, ndim=3] img_index,
                         np.ndarray[DTYPE_t, ndim=2] flat_out):
    """
    imgs has shape (n_filters, n_channels, img_h, img_w)
    win_shape has shape (win_h, win_w)
    img_index has shape (n_filters, img_h, img_w)
    flat_out has shape ((img_h * img_w * n_filters), (n_channels * win_h * win_w))
    """
    cdef uint frag = imgs.shape[0]
    cdef uint img_channels = imgs.shape[1]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]

    cdef uint win_h = win_shape[0]
    cdef uint win_w = win_shape[1]

    cdef uint seg_mid_h = win_h // 2
    cdef uint seg_mid_w = win_w // 2
    cdef uint mid_off_h = win_h % 2
    cdef uint mid_off_w = win_w % 2

    cdef uint fg, org_index, y, x, win_y, win_x, img_y, img_x, ch, counter
    cdef int yMin, yMax, xMin, xMax

    for fg in range(frag):
        for y in range(img_h):
            for x in range(img_w):
                org_index = <uint>img_index[fg, y, x]
                counter = 0

                yMin = y-seg_mid_h
                yMax = y+seg_mid_h+mid_off_h
                for win_y in range(yMin, yMax):
                    img_y = getImgIndex(win_y, img_h)
                    xMin = x-seg_mid_w
                    xMax = x+seg_mid_w+mid_off_w
                    for win_x in range(xMin, xMax):
                        img_x = getImgIndex(win_x, img_w)

                        for ch in range(img_channels):
                            flat_out[org_index, counter] = imgs[fg, ch, img_y, img_x]
                            counter+=1
    return flat_out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline getImgIndex(int tempIndex, uint size):
    cdef uint index
    if(tempIndex < 0):        
        index = <uint>((tempIndex * -1) % size)
    elif(tempIndex >= size):
        index = <uint>(size - (tempIndex % size) - 1)
    else:
        index = <uint>(tempIndex)
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
def flatten_seg_bc01_bprop(np.ndarray[DTYPE_t, ndim=2] flat_grade,
                         tuple win_shape,
                         np.ndarray[long, ndim=3] img_index,
                         np.ndarray[DTYPE_t, ndim=4] imgs_grad):
    """
    imgs_grad has shape (n_filters, n_channels, img_h, img_w)
    win_shape has shape (win_h, win_w)
    img_index has shape (n_filters, img_h, img_w)
    flat_grade has shape ((img_h * img_w * n_filters), (n_channels * win_h * win_w))
    """
    cdef uint frag = imgs_grad.shape[0]
    cdef uint img_channels = imgs_grad.shape[1]
    cdef uint img_h = imgs_grad.shape[2]
    cdef uint img_w = imgs_grad.shape[3]

    cdef uint win_h = win_shape[0]
    cdef uint win_w = win_shape[1]

    cdef uint seg_mid_h = win_h // 2
    cdef uint seg_mid_w = win_w // 2
    cdef uint mid_off_h = win_h % 2
    cdef uint mid_off_w = win_w % 2

    cdef uint fg, org_index, y, x, win_y, win_x, img_y, img_x, ch, counter
    cdef int yMin, yMax, xMin, xMax

    for fg in range(frag):
        for y in range(img_h):
            for x in range(img_w):
                if img_index[fg, y, x] < 0: 
                    continue
                org_index = <uint>img_index[fg, y, x]
                counter = 0
                yMin = y-seg_mid_h
                yMax = y+seg_mid_h+mid_off_h
                for win_y in range(yMin, yMax):
                    img_y = getImgIndex(win_y, img_h)
                    xMin = x-seg_mid_w
                    xMax = x+seg_mid_w+mid_off_w
                    for win_x in range(xMin, xMax):
                        img_x = getImgIndex(win_x, img_w)

                        for ch in range(img_channels):
                            imgs_grad[fg, ch, img_y, img_x] += flat_grade[org_index, counter]
                            counter+=1
    return imgs_grad
    