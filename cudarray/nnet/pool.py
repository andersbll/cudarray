import numpy as np
import cudarray as ca
from ..cudarray_wrap import nnet as wrap


class PoolB01(object):
    def __init__(self, win_shape, padding, strides, method='max'):
        self.win_shape = win_shape
        self.padding = padding
        self.strides = strides
        if method not in ['max']:
            raise ValueError('invalid pooling method')
        self.mask = None

    def fprop(self, imgs, poolout=None):
        poolout_shape = self.output_shape(imgs.shape)
        if poolout is None:
            poolout = ca.empty(poolout_shape, dtype=imgs.dtype)
        else:
            if poolout_shape != poolout.shape:
                raise ValueError('poolout.shape does not match result')
            if imgs.dtype != poolout.dtype:
                raise ValueError('dtype mismatch')

        if self.mask is None or self.mask.shape != poolout_shape:
            self.mask = ca.empty(poolout_shape, dtype=np.dtype('int32'))

        img_shape = imgs.shape[-2:]
        n_imgs = np.prod(imgs.shape[:-2])
        wrap._max_pool_b01(
            imgs._data, n_imgs, img_shape, self.win_shape, self.padding,
            self.strides, poolout._data, self.mask._data
        )
        return poolout

    def bprop(self, img_shape, poolout_d, imgs_d=None):
        n_imgs_shape = poolout_d.shape[:-2]
        imgs_shape = n_imgs_shape + img_shape

        if imgs_d is None:
            imgs_d = ca.empty(imgs_shape, dtype=poolout_d.dtype)
        else:
            if imgs_d.shape != imgs_d_shape:
                raise ValueError('poolout.shape does not match result')
            if imgs_d.dtype != poolout_d.dtype:
                raise ValueError('dtype mismatch')

        n_imgs = np.prod(n_imgs_shape)
        wrap._max_pool_b01_bprop(
            poolout_d._data, self.mask._data, n_imgs, img_shape,
            self.win_shape, self.padding, self.strides, imgs_d._data
        )
        return imgs_d

    def output_shape(self, imgs_shape):
        n_imgs_shape = imgs_shape[:-2]
        img_h, img_w = imgs_shape[-2:]
        out_shape = ((img_h + 2*self.padding[0] - self.win_shape[0])
                     / self.strides[0] + 1,
                     (img_w + 2*self.padding[1] - self.win_shape[1])
                     / self.strides[1] + 1)
        return n_imgs_shape + out_shape
