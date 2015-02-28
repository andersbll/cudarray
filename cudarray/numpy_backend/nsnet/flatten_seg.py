from .flatten_seg_bc01 import *
import cudarray as ca


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
            flatout = ca.empty(flatout_shape, dtype=imgs.dtype)
        else:
            if flatout.shape != flatout_shape:
                raise ValueError('convout.shape does not match result')
            if flatout.dtype != imgs.dtype:
                raise ValueError('dtype mismatch')

        flatout = flatten_seg_bc01(imgs=imgs,
                                   win_shape=self.win_shape,
                                   img_index=self.input_index,
                                   flat_out=flatout)

        return flatout

    def bprop(self, flatout_d, imgs_d=None):

        if imgs_d is None:
            imgs_d = ca.zeros(self.img_shape, dtype=flatout_d.dtype)
        else:
            if self.img_shape != imgs_d.shape:
                raise ValueError('imag_d.shape does not match imgs_d')

        flatten_seg_bc01_bprop(flat_grade = flatout_d,
                               win_shape = self.win_shape,
                               img_index = self.input_index,
                               imgs_grad = imgs_d)

        return imgs_d

    def output_index(self, input_index, output_index=None):
        self.input_index = input_index
        return output_index

    def output_shape(self, imgs_shape):
        n_filters, n_channels, img_h, img_w = imgs_shape
        outIndex = self.input_index + 1
        win_h, win_w = self.win_shape
        return (ca.count_nonzero(outIndex), (n_channels * win_h * win_w))
