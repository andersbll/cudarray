import numpy as np
import cudarray as ca
from .pool_bc01 import *


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
            poolout = np.zeros(poolout_shape, dtype=imgs.dtype)
        else:
            if poolout_shape != poolout.shape:
                raise ValueError('poolout.shape does not match result')
            if imgs.dtype != poolout.dtype:
                raise ValueError('dtype mismatch')

        img_shape = imgs.shape[-2:]
        n_imgs = np.prod(imgs.shape[:-2])
        switches = None
        pool_bc01(imgs = imgs,
              win_shape = (2, 2),
              strides = (2, 2),
              poolout = poolout,
              switches = switches)

        self.mask = switches
        print ("BACK")
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

        bprop_pool_bc01(poolout_grad = poolout_d,
                    win_shape = self.win_shape,
                    strides = self.strides,
                    switches = self.mask,
                    imgs_grad = imgs_d)
        return imgs_d

    def output_shape(self, imgs_shape):
        n_imgs_shape = imgs_shape[:-2]
        img_h, img_w = imgs_shape[-2:]
        out_shape = ((img_h + 2*self.padding[0] - self.win_shape[0])
                     / self.strides[0] + 1,
                     (img_w + 2*self.padding[1] - self.win_shape[1])
                     / self.strides[1] + 1)
        return n_imgs_shape + out_shape
