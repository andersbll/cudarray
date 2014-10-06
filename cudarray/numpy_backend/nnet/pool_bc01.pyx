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


@cython.boundscheck(False)
@cython.wraparound(False)
def pool_bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
              tuple win_shape,
              tuple strides,
              uint type = POOL_MAX,
              np.ndarray[DTYPE_t, ndim=4] poolout = None,
              np.ndarray[np.int_t, ndim=5] switches = None):
    """ Multi-image, multi-channel pooling
    imgs has shape (n_imgs, n_channels, img_h, img_w)
    win_shape has shape (win_h, win_w) 
    strides has shape (stride_y, stride_x)
    poolout has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x)
    switches has shape (n_imgs, n_channels, img_h//stride_y, img_w//stride_x, 2)
    """

    cdef uint pool_h = win_shape[0] 
    cdef uint pool_w = win_shape[1]
    cdef uint pool_size = pool_h * pool_w
    cdef uint stride_x = strides[1] 
    cdef uint stride_y = strides[0] 

    cdef uint n_imgs = imgs.shape[0]
    cdef uint n_channels = imgs.shape[1]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]

    cdef uint out_h = img_h // stride_y
    cdef uint out_w = img_w // stride_x

    cdef int pool_h_top = pool_h // 2 - 1 + pool_h % 2
    cdef int pool_h_bottom = pool_h // 2 + 1
    cdef int pool_w_left = pool_w // 2 - 1 + pool_w % 2
    cdef int pool_w_right = pool_w // 2 + 1

    if (poolout == None):
        poolout = np.empty(shape=(n_imgs, n_channels, img_h//stride_y, img_w//stride_x),
                           dtype=DTYPE)

    if (switches == None):
        switches = np.empty(shape=(n_imgs, n_channels, img_h//stride_y, img_w//stride_x, 2),
                           dtype=DTYPE)


    if not n_imgs == poolout.shape[0] == switches.shape[0]:
        raise ValueError('Mismatch in number of images.')
    if not n_channels == poolout.shape[1] == switches.shape[1]:
        raise ValueError('Mismatch in number of channels.')
    if not (out_h == poolout.shape[2] == switches.shape[2] and out_w == poolout.shape[3] == switches.shape[3]):
        raise ValueError('Mismatch in image shape.')
    if not switches.shape[4] == 2:
        raise ValueError('switches should only have length 2 in the 5. dimension.')


    cdef uint i, c, y, x, y_out, x_out
    cdef int y_min, y_max, x_min, x_max
    cdef uint img_y, img_x
    cdef uint img_y_max = 0
    cdef uint img_x_max = 0
    cdef DTYPE_t value, new_value

    for i in range(n_imgs):
        for c in range(n_channels):
            for y_out in range(out_h):
                y = y_out*stride_y
                y_min = int_max(y-pool_h_top, 0)
                y_max = int_min(y+pool_h_bottom, img_h)
                for x_out in range(out_w):
                    x = x_out*stride_x
                    x_min = int_max(x-pool_w_left, 0)
                    x_max = int_min(x+pool_w_right, img_w)
                    if (type == POOL_MAX):
                        value = -9e99
                    else: 
                        value = 0

                    for img_y in range(y_min, y_max):
                        for img_x in range(x_min, x_max):
                            if (type == POOL_MAX):
                                new_value = imgs[i, c, img_y, img_x]
                                if new_value > value:
                                    value = new_value
                                    img_y_max = img_y
                                    img_x_max = img_x
                            else:
                                value += imgs[i, c, img_y, img_x]
                    if (type == POOL_MAX):
                        poolout[i, c, y_out, x_out] = value
                        switches[i, c, y_out, x_out, 0] = img_y_max
                        switches[i, c, y_out, x_out, 1] = img_x_max
                    else:
                        poolout[i, c, y_out, x_out] = value / pool_size


@cython.boundscheck(False)
@cython.wraparound(False)
def bprop_pool_bc01(np.ndarray[DTYPE_t, ndim=4] poolout_grad,
                    tuple win_shape,
                    tuple strides,
                    uint type = POOL_MAX,
                    np.ndarray[np.int_t, ndim=5] switches = None,
                    np.ndarray[DTYPE_t, ndim=4] imgs_grad = None):

    cdef uint n_imgs = poolout_grad.shape[0]
    cdef uint n_channels = poolout_grad.shape[1]
    cdef uint poolout_h = poolout_grad.shape[2]
    cdef uint poolout_w = poolout_grad.shape[3]

    cdef uint pool_h = win_shape[0] 
    cdef uint pool_w = win_shape[1]
    cdef uint pool_size = pool_h * pool_w
    cdef uint stride_x = strides[1] 
    cdef uint stride_y = strides[0] 

    cdef uint i, c, y, x, img_y, img_y_min, img_x_min, img_y_max, img_x_max

    if (type == POOL_MAX and switches == None):
        raise ValueError('switches has to be defined')

    if (imgs_grad == None):
        imgs_grad = np.zeros(shape=(n_imgs, n_channels, poolout_h*stride_y, poolout_w*stride_x),
                             dtype=DTYPE)
    else:
        imgs_grad[...] = 0

    for i in range(n_imgs):
        for c in range(n_channels):
            for y in range(poolout_h):
                for x in range(poolout_w):
                    if (type == POOL_MEAN):
                        img_y_min = y * stride_y
                        img_x_min = x * stride_x
                        img_y_max = img_y_min + pool_h - 1
                        img_x_max = img_x_min + pool_w - 1
                        # XXX should be += instead of =
                        imgs_grad[i, c, img_y_min : img_y_max, img_x_min : img_x_max] += (poolout_grad[i, c, y, x] / pool_size)
                    elif (type == POOL_MAX):
                        img_y = switches[i, c, y, x, 0]
                        img_x = switches[i, c, y, x, 1]
                        # XXX should be += instead of =
                        imgs_grad[i, c, img_y, img_x] += poolout_grad[i, c, y, x]
    return imgs_grad
