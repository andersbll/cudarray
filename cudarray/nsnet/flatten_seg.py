import cudarray as ca
import numpy as np
from ..numpy_backend import flatten_seg_bc01_bprop
from ..numpy_backend import flatten_seg_bc01


class FlattenBC01(object):
    def __init__(self, win_shape=(1,1)):
        self.padding = 'mirror'
        self.win_shape = win_shape
        self.img_shape = None
        self.input_index = None

    def fprop(self, imgs, flatout=None):
        self.img_shape = imgs.shape

        flatout_shape = self.output_shape(self.img_shape)
        if flatout is None:
            flatout = np.empty(flatout_shape, dtype=np.float)
        else:
            if flatout.shape != flatout_shape:
                raise ValueError('convout.shape does not match result')
            if flatout.dtype != imgs.dtype:
                raise ValueError('dtype mismatch')

        imgs = np.array(imgs)
        imgs = imgs.astype(np.float)
        flatout = flatten_seg_bc01(imgs=imgs,
                                   win_shape=self.win_shape,
                                   img_index=self.input_index,
                                   flat_out=flatout)

        return ca.array(flatout)

    def bprop(self, flatout_d, imgs_d=None):

        if imgs_d is None:
            imgs_d = np.empty(self.img_shape, dtype=np.float)
        else:
            if self.img_shape != imgs_d.shape:
                raise ValueError('imag_d.shape does not match imgs_d')

        flatout_d = np.array(flatout_d)
        flatout_d = flatout_d.astype(np.float)
        flatten_seg_bc01_bprop(flat_grade = flatout_d,
                               win_shape = self.win_shape,
                               img_index = self.input_index,
                               imgs_grad = imgs_d)

        return ca.array(imgs_d)

    def output_index(self, input_index, output_index=None):
        self.input_index = input_index
        return output_index

    def output_shape(self, imgs_shape):
        n_filters, n_channels, img_h, img_w = imgs_shape
        outIndex = self.input_index + 1
        win_h, win_w = self.win_shape
        return (np.count_nonzero(outIndex), (n_channels * win_h * win_w))
