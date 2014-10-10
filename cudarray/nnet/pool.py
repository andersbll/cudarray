import numpy as np
import cudarray as ca
from ..cudarray_wrap import nnet as wrap


def pool_b01(imgs, win_shape, padding, strides, method='max', poolout=None,
             mask=None):
    img_h, img_w = imgs.shape[-2:]
    n_imgs_shape = imgs.shape[:-2]
    poolout_h = (img_h + 2*padding[0] - win_shape[0])/strides[0] + 1
    poolout_w = (img_w + 2*padding[1] - win_shape[1])/strides[1] + 1
    poolout_shape = n_imgs_shape + (poolout_h, poolout_w)

    if poolout is None:
        poolout = ca.empty(poolout_shape, dtype=imgs.dtype)
    else:
        if poolout_shape != poolout.shape:
            raise ValueError('poolout.shape does not match result')
        if imgs.dtype != poolout.dtype:
            raise ValueError('dtype mismatch')

    if mask is None:
        mask = ca.empty(poolout_shape, dtype=np.dtype('int32'))
    else:
        if mask.shape != poolout.shape:
            raise ValueError('poolout.shape does not match result')
        if mask.dtype != np.dtype('int32'):
            raise ValueError('dtype mismatch')

    n_imgs = np.prod(n_imgs_shape)
    wrap._max_pool_b01(imgs._data, n_imgs, (img_h, img_w), win_shape, padding,
                       strides, poolout._data, mask._data)
    return poolout, mask


def pool_b01_bprop(poolout_d, mask, img_shape, win_shape, padding, strides,
                   method='max', imgs_d=None):
    img_h, img_w = img_shape
    n_imgs_shape = poolout_d.shape[:-2]
    imgs_shape = n_imgs_shape + (img_h, img_w)

    if imgs_d is None:
        imgs_d = ca.empty(imgs_shape, dtype=poolout_d.dtype)
    else:
        if imgs_d.shape != imgs_d_shape:
            raise ValueError('poolout.shape does not match result')
        if imgs_d.dtype != poolout_d.dtype:
            raise ValueError('dtype mismatch')

    n_imgs = np.prod(n_imgs_shape)
    wrap._max_pool_b01_bprop(poolout_d._data, mask._data, n_imgs, img_shape,
                             win_shape, padding, strides, imgs_d._data)
    return imgs_d
