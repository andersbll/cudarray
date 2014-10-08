from __future__ import division
import numpy as np
import cython
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint


@cython.boundscheck(False)
@cython.wraparound(False)
def lrnorm_bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
              uint N,
              DTYPE_t alpha,
              DTYPE_t beta,
              DTYPE_t k):
    """
    imgs has shape (n_imgs, n_channels, img_h, img_w)
    TODO: Optimer til cython : 
    ('n_train ', 200)
    ('n_valid ', 75)
    ('n_test ', 150)
    ('n_train ', 200)
    ('n_valid ', 75)
    ('n_test ', 150)
    ncalls  tottime  percall  cumtime  percall 
    85      9.543    0.112    9.543    0.112 
    """

    cdef DTYPE_t norm_window 
    cdef uint n_imgs = imgs.shape[0]
    cdef uint n_channels = imgs.shape[1]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]

    cdef uint half = N // 2
    cdef uint tailLength = N - half

    cdef int max_channel

    cdef DTYPE_t a_i
    cdef DTYPE_t a_half

    tail = tailLength*[0.0]

    for i in range(n_imgs):
        for y in range(img_h):
            for x in range(img_w):
                norm_window = 0.0
                tail = tailLength*[0.0]

                for a in range(N + 1):
                    norm_window += (imgs[i, a, y, x] ** 2)

                for c in range(n_channels):
                    a_i = imgs[i, c, y, x]
                    a_half = tail.pop(0)
                    tail.append(a_i)
                    #Normalazation 
                    imgs[i, c, y, x] = a_i / ((k + alpha * norm_window) ** beta)
                    #Move the window for next channel
                    max_channel = half + c + 1
                    #Move window if possible 
                    if (max_channel < n_channels and (c >= N) ):
                        norm_window += (imgs[i, max_channel, y, x] ** 2)
                        #Remove privius channel from sum
                        norm_window -= (a_half ** 2)

    return imgs
