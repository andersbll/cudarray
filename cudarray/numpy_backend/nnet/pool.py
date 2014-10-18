import numpy as np
import cudarray as ca
from .pool_bc01 import *


class PoolB01(object):
    def __init__(self, win_shape, padding, strides, method='max'):
        self.win_shape = win_shape
        self.padding = padding
        self.strides = strides
        if method not in ['max', 'men']:
            raise ValueError('invalid pooling method')
        if method == 'max':
            self.method = 0
        elif method == 'men':
            self.method = 1

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

        if self.mask is None or self.mask.shape[:-1] != poolout_shape:
            self.mask = ca.empty(poolout_shape + (2,), dtype=np.dtype('int_'))

        pool_bc01(imgs=imgs,
                  win_shape=self.win_shape,
                  strides=self.strides,
                  padding=self.padding,
                  poolout=poolout,
                  type=self.method,
                  switches=self.mask)

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

        bprop_pool_bc01(poolout_grad=poolout_d,
                        win_shape=self.win_shape,
                        strides=self.strides,
                        padding=self.padding,
                        type=self.method,
                        switches=self.mask,
                        imgs_grad=imgs_d)
        return imgs_d

    def output_shape(self, imgs_shape):
        n_imgs_shape = imgs_shape[:-2]
        img_h, img_w = imgs_shape[-2:]
        out_shape = ((img_h + 2*self.padding[0] - self.win_shape[0])
                     / self.strides[0] + 1,
                     (img_w + 2*self.padding[1] - self.win_shape[1])
                     / self.strides[1] + 1)
        return n_imgs_shape + out_shape
