from .conv_seg_bc01 import *
import cudarray as ca



class ConvBC01(object):
    def __init__(self):
        self.type = 'max'

    def fprop(self, imgs, filters, convout=None):
        fg, c, img_h, img_w = imgs.shape
        f, c_filters, filter_h, filter_w = filters.shape

        if c != c_filters:
            raise ValueError('channel mismatch')
        if imgs.dtype != filters.dtype:
            raise ValueError('dtype mismatch')

        convout_shape = self.output_shape(imgs.shape, f)
        if convout is None:
            convout = ca.empty(convout_shape, dtype=imgs.dtype)
        else:
            if convout.shape != convout_shape:
                raise ValueError('convout.shape does not match result')
            if convout.dtype != imgs.dtype:
                raise ValueError('dtype mismatch')

        convout = conv_seg_bc01(imgs=imgs,
                      filters=filters,
                      convout=convout)

        return convout

    def bprop(self, imgs, filters, convout_d, to_filters=True, to_imgs=True,
              filters_d=None, imgs_d=None):
        fg, c, _, _ = imgs.shape
        f, c_filters, _, _ = filters.shape
        fg_convout, f_convout, _, _ = convout_d.shape

        if f != f_convout:
            raise ValueError('filter mismatch')
        if c != c_filters:
            raise ValueError('channel mismatch')
        if fg != fg_convout:
            raise ValueError('fragment mismatch')

        if imgs.dtype != filters.dtype != convout_d.dtype:
            raise ValueError('dtype mismatch')

        if filters_d is None:
            filters_d = ca.empty(filters.shape, dtype=filters.dtype)

        if imgs_d is None:
            imgs_d = ca.empty(imgs.shape, dtype=imgs.dtype)

        conv_seg_bc01_bprop(imgs=imgs,
                            convout_d=convout_d,
                            filters=filters,
                            imgs_grad=imgs_d,
                            filters_grad=filters_d)

        return filters_d, imgs_d

    def output_shape(self, imgs_shape, n_filters):
        fg, _, img_h, img_w = imgs_shape
        return (fg, n_filters, img_h, img_w)
