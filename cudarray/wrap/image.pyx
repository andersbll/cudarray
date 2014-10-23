cimport numpy as np

cimport image
from .array_data cimport ArrayData


def _img2win(ArrayData imgs, int n_imgs, img_shape, win_shape, padding,
             strides, ArrayData wins):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int win_h = win_shape[0]
    cdef int win_w = win_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]

    if imgs.dtype == np.dtype('float32'):
        image.img2win(<float *>imgs.dev_ptr, n_imgs, img_h, img_w, win_h,
            win_w, pad_y, pad_x, stride_y, stride_x, <float *>wins.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(imgs.dtype))

def _win2img(ArrayData wins, int n_imgs, img_shape, win_shape, padding,
             strides, ArrayData imgs):
    cdef int img_h = img_shape[0]
    cdef int img_w = img_shape[1]
    cdef int win_h = win_shape[0]
    cdef int win_w = win_shape[1]
    cdef int pad_y = padding[0]
    cdef int pad_x = padding[1]
    cdef int stride_y = strides[0]
    cdef int stride_x = strides[1]

    if imgs.dtype == np.dtype('float32'):
        image.img2win(<float *>wins.dev_ptr, n_imgs, img_h, img_w, win_h,
            win_w, pad_y, pad_x, stride_y, stride_x, <float *>imgs.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(imgs.dtype))
