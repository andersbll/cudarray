from .conv_bc01 import *
import cudarray as ca


class ConvBC01(object):
    def __init__(self, padding, strides):
        self.padding = padding
        self.strides = strides

    def fprop(self, imgs, filters, convout=None):
        b, c, img_h, img_w = imgs.shape
        f, c_filters, filter_h, filter_w = filters.shape
        if c != c_filters:
            raise ValueError('channel mismatch')
        if imgs.dtype != filters.dtype:
            raise ValueError('dtype mismatch')

        convout_shape = self.output_shape(imgs.shape, f, (filter_h, filter_w))
        if convout is None:
            convout = ca.empty(convout_shape, dtype=imgs.dtype)
        else:
            if convout.shape != convout_shape:
                raise ValueError('convout.shape does not match result')
            if convout.dtype != imgs.dtype:
                raise ValueError('dtype mismatch')

        conv_bc01(imgs=imgs,
                  filters=filters,
                  padding=self.padding,
                  strides=self.strides,
                  convout=convout)

        self.last_imgs = imgs

        return convout

    def bprop(self, imgs, filters, convout_d, to_filters=True, to_imgs=True,
              filters_d=None, imgs_d=None):
        if imgs is None:
            imgs = self.last_imgs
        b, c, _, _ = imgs.shape
        f, c_filters, _, _ = filters.shape
        b_convout, f_convout, _, _ = convout_d.shape

        if b != b_convout:
            raise ValueError('batch mismatch')
        if f != f_convout:
            raise ValueError('filter mismatch')
        if c != c_filters:
            raise ValueError('channel mismatch')

        if imgs.dtype != filters.dtype != convout_d.dtype:
            raise ValueError('dtype mismatch')

        if filters_d is None:
            filters_d = ca.empty(filters.shape, dtype=filters.dtype)

        if imgs_d is None:
            imgs_d = ca.empty(imgs.shape, dtype=imgs.dtype)

        conv_bc01_bprop(imgs=imgs,
                        convout_d=convout_d,
                        filters=filters,
                        padding=self.padding,
                        strides=self.strides,
                        imgs_grad=imgs_d,
                        filters_grad=filters_d)

        return filters_d, imgs_d

    def output_shape(self, imgs_shape, n_filters, filter_shape):
        b, _, img_h, img_w = imgs_shape
        out_shape = ((img_h + 2*self.padding[0] - filter_shape[0])
                     / self.strides[0] + 1,
                     (img_w + 2*self.padding[1] - filter_shape[1])
                     / self.strides[1] + 1)
        return (b, n_filters) + out_shape
