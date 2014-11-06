import numpy as np
import cudarray as ca
from .pool_seg_bc01 import *


class PoolB01(object):
    def __init__(self, win_shape):
        self.win_shape = win_shape
        self.mpIDXS = None
        self.img_shape = None

    def fprop(self, imgs, poolout=None):
        if self.img_shape is None:
            self.img_shape = imgs.shape

        poolout_shape = self.output_shape(imgs.shape)
        if poolout is None:
            poolout = ca.empty(poolout_shape, dtype=imgs.dtype)
        else:
            if poolout_shape != poolout.shape:
                raise ValueError('poolout.shape does not match result')
            if imgs.dtype != poolout.dtype:
                raise ValueError('dtype mismatch')

        if self.mpIDXS is None or self.mpIDXS.shape[:-1] != poolout_shape:
            self.mpIDXS = ca.empty(poolout_shape + (3,), dtype=np.dtype('int_'))

        pool_seg_max_bc01(imgs=imgs,
                          win_shape=self.win_shape,
                          poolout=poolout,
                          switches=self.mpIDXS)

        return poolout

    def bprop(self, poolout_d, imgs_d=None):
        if imgs_d is None:
            imgs_d = ca.empty(self.img_shape, dtype=poolout_d.dtype)
        else:
            if imgs_d.shape != imgs_d_shape:
                raise ValueError('poolout.shape does not match result')
            if imgs_d.dtype != poolout_d.dtype:
                raise ValueError('dtype mismatch')

        bprop_pool_seg_bc01(poolout_grad=poolout_d,
                            win_shape=self.win_shape,
                            switches=self.mpIDXS,
                            imgs_grad=imgs_d)
        return imgs_d

    def output_index(self, input_index, output_index=None):
        if output_index == None:
            f_out = input_index.shape[0] * self.win_shape[0] * self.win_shape[1] 
            index_h, index_w = input_index.shape[-2:]
            out_shape = ((index_h // self.win_shape[0]) + (index_h % self.win_shape[0] > 0),
                     (index_w // self.win_shape[1]) + (index_w % self.win_shape[1] > 0))
            
            output_index = ca.empty(((f_out,)+out_shape), dtype=input_index.dtype)

        pool_seg_indexing_bc01(imgs=input_index,
                               win_shape=self.win_shape,
                               poolout=output_index)

        return output_index

    def output_shape(self, imgs_shape):
        f_in = imgs_shape[0]
        c_in = imgs_shape[1]
        f_out = f_in * self.win_shape[0] * self.win_shape[1] 
        img_h, img_w = imgs_shape[-2:]
        out_shape = ((img_h // self.win_shape[0]) + (img_h % self.win_shape[0] > 0),
                     (img_w // self.win_shape[1]) + (img_w % self.win_shape[1] > 0))
        return (f_out, c_in) + out_shape
