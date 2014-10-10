import cudarray as ca
from ..cudarray_wrap import nnet as wrap


def conv_bc01(imgs, filters, padding, strides, convout=None):
    b, c, img_h, img_w = imgs.shape
    f, c_filters, filter_h, filter_w = filters.shape

    convout_h = (img_h + 2 * padding[0] - filter_h) / strides[0] + 1
    convout_w = (img_w + 2 * padding[1] - filter_w) / strides[1] + 1
    convout_shape = (b, f, convout_h, convout_w)

    if c != c_filters:
        raise ValueError('channel mismatch')

    if imgs.dtype != filters.dtype:
        raise ValueError('dtype mismatch')

    if convout is None:
        convout = ca.empty(convout_shape, dtype=imgs.dtype)
    else:
        if convout.shape != convout_shape:
            raise ValueError('convout.shape does not match result')
        if convout.dtype != imgs.dtype:
            raise ValueError('dtype mismatch')

    wrap._conv_bc01_matmul(
        imgs._data, filters._data, b, c, f, (img_h, img_w),
        (filter_h, filter_w), padding, strides, convout._data
    )
    return convout


def conv_bc01_bprop_filters(imgs, convout_d, filter_shape, padding, strides,
                            filters_d=None):
    b, c, img_h, img_w = imgs.shape
    b_convout, f, convout_h, convout_w = convout_d.shape
    filters_shape = (f, c) + filter_shape

    if b != b_convout:
        raise ValueError('batch mismatch')

    if imgs.dtype != convout_d.dtype:
        raise ValueError('dtype mismatch')

    if filters_d is None:
        filters_d = ca.empty(filters_shape, dtype=imgs.dtype)
    else:
        if filters_d.shape != filters_shape:
            raise ValueError('filters_d.shape does not match result')
        if filters_d.dtype != imgs.dtype:
            raise ValueError('dtype mismatch')

    wrap._conv_bc01_matmul_bprop_filters(
        imgs._data, convout_d._data, b, c, f, (img_h, img_w), filter_shape,
        padding, strides, filters_d._data
    )
    return filters_d


def conv_bc01_bprop_imgs(filters, convout_d, img_shape, padding, strides,
                         imgs_d=None):
    f, c, filter_h, filter_w = filters.shape
    b, convout_f, convout_h, convout_w = convout_d.shape
    imgs_shape = (b, c,) + img_shape

    if f != convout_f:
        raise ValueError('channel mismatch')

    if filters.dtype != convout_d.dtype:
        raise ValueError('dtype mismatch')

    if imgs_d is None:
        imgs_d = ca.empty(imgs_shape, dtype=filters.dtype)
    else:
        if imgs_d.shape != imgs_shape:
            raise ValueError('imgs_d.shape does not match result')
        if imgs_d.dtype != filters.dtype:
            raise ValueError('dtype mismatch')

    wrap._conv_bc01_matmul_bprop_imgs(
        filters._data, convout_d._data, b, c, f, img_shape,
        (filter_h, filter_w), padding, strides, imgs_d._data
    )
    return imgs_d
