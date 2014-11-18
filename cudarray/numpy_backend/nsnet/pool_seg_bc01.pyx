from __future__ import division
import numpy as np
import cython
cimport numpy as np

cdef int POOL_MAX = 0
cdef int POOL_MEAN = 1

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint

cdef inline DTYPE_t dtype_t_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b


def pool_seg_max_bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
              tuple win_shape,
              tuple strides,
              np.ndarray[DTYPE_t, ndim=4] poolout,
              np.ndarray[np.int_t, ndim=5] switches):
    """ Multi-image, multi-channel pooling
    imgs has shape (n_imgs, n_channels, img_h, img_w)
    win_shape has shape (win_h, win_w) 
    strides has shape (stride_y, stride_x)
    poolout has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x)
    switches has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x, 2)
    """
    cdef uint pool_h = win_shape[0] 
    cdef uint pool_w = win_shape[1]
    cdef uint stride_h = strides[0]
    cdef uint stride_w = strides[1]

    cdef uint F_in = imgs.shape[0]
    cdef uint n_channels = imgs.shape[1]

    cdef uint out_h = poolout.shape[2]
    cdef uint out_w = poolout.shape[3]

    cdef uint F_out_local = pool_h * pool_w

    cdef uint i, c, y, x, y_out, x_out, fg_out, img_y_max, img_x_max, y_frag, x_frag
    cdef DTYPE_t value

    for fg_in in range(F_in):
        for c in range(n_channels):
            for y_out in range(out_h):
                y = y_out*stride_h
                for x_out in range(out_w):
                    x = x_out*stride_w

                    f_count = 0
                    for off_y in range(pool_h):
                        for off_x in range(pool_w):
                            y_frag = y + off_y
                            x_frag = x + off_x
                            fg_out = fg_in * F_out_local + f_count
                            #Get the value, and position of the max in pool win
                            value, img_y_max, img_x_max = max_value(fg_in, c, y_frag, x_frag, pool_h, pool_w, imgs)
                            poolout[fg_out, c, y_out, x_out] = value
                            switches[fg_out, c, y_out, x_out, 0] = img_y_max
                            switches[fg_out, c, y_out, x_out, 1] = img_x_max
                            switches[fg_out, c, y_out, x_out, 2] = fg_in

                            f_count += 1

cdef inline max_value(uint fg, uint c, uint y_start,
                      uint x_start, uint pool_h, uint pool_w,
                      np.ndarray[DTYPE_t, ndim=4] imgs):

    cdef DTYPE_t value = -9e99
    cdef DTYPE_t new_value

    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]
    cdef uint img_y_max, img_x_max, img_x, img_y

    y_min = y_start
    y_max = int_min(y_start+pool_h, img_h)
    x_min = x_start
    x_max = int_min(x_start+pool_w, img_w)

    #print (str(x_start) + " :: " + str(x_min) + " , " + str(x_max))
    #print (str(y_start) + " :: " + str(y_min) + " , " + str(y_max))

    for img_y in range(y_min, y_max):
        for img_x in range(x_min, x_max):
            new_value = imgs[fg, c, img_y, img_x]
            if new_value > value:
                value = new_value
                img_y_max = img_y
                img_x_max = img_x
    return value, img_y_max, img_x_max


def bprop_pool_seg_bc01(np.ndarray[DTYPE_t, ndim=4] poolout_grad,
                    np.ndarray[np.int_t, ndim=5] switches,
                    np.ndarray[DTYPE_t, ndim=4] imgs_grad):

    cdef uint F_out = poolout_grad.shape[0]
    cdef uint n_channels = poolout_grad.shape[1]
    cdef uint poolout_h = poolout_grad.shape[2]
    cdef uint poolout_w = poolout_grad.shape[3]

    cdef uint i, c, y, x, img_y, img_x, fg_in, fg

    imgs_grad[...] = 0

    for fg in range(F_out):
        for c in range(n_channels):
            for y in range(poolout_h):
                for x in range(poolout_w):
                    img_y = switches[fg, c, y, x, 0]
                    img_x = switches[fg, c, y, x, 1]
                    fg_in = switches[fg, c, y, x, 2]
                    # XXX should be += instead of =
                    imgs_grad[fg_in, c, img_y, img_x] += poolout_grad[fg, c, y, x]
    return imgs_grad

def pool_seg_indexing_bc01(np.ndarray[DTYPE_t, ndim=3] imgs,
                           tuple win_shape,
                           tuple strides,
                           np.ndarray[DTYPE_t, ndim=3] poolout):
    """ Multi-image, multi-channel pooling
    imgs has shape (n_imgs, n_channels, img_h, img_w)
    win_shape has shape (win_h, win_w) 
    strides has shape (stride_y, stride_x)
    poolout has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x)
    switches has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x, 2)
    """
    cdef uint pool_h = win_shape[0] 
    cdef uint pool_w = win_shape[1]
    cdef uint stride_h = strides[0] 
    cdef uint stride_w = strides[1]

    cdef uint F_in = imgs.shape[0]

    cdef uint out_h = poolout.shape[1]
    cdef uint out_w = poolout.shape[2]

    cdef uint F_out_local = pool_h * pool_w

    cdef uint i, y, x, y_out, x_out, fg_out, y_frag, x_frag

    for fg_in in range(F_in):
        for y_out in range(out_h):
            y = y_out*stride_h
            for x_out in range(out_w):
                x = x_out*stride_w

                f_count = 0
                for off_y in range(pool_h):
                    for off_x in range(pool_w):
                        y_frag = y + off_y
                        x_frag = x + off_x
                        fg_out = fg_in * F_out_local + f_count
                        poolout[fg_out, y_out, x_out] = imgs[fg_in, y_frag, x_frag]
                        f_count += 1
