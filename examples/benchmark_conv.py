#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.weight_acts import WeightActs
from pylearn2.sandbox.cuda_convnet.img_acts import ImageActs
from theano_ops.conv_bc01_fft import (ConvBC01, ConvBC01ImgsGrad,
                                      ConvBC01FiltersGrad)
import cudarray as ca


def avg_running_time(fun):
    n_iter = 20
    start_time = time.time()
    for _ in range(n_iter):
        fun()
    duration = time.time() - start_time
    return duration / float(n_iter)


def allclose(a, b):
    atol = 1e-3
    rtol = 1e-3
    return np.allclose(a, b, atol=atol, rtol=rtol)


def benchmark(n_imgs, n_channels, img_shape, n_filters, filter_shape, pad):
    print('\nn_imgs: %i, n_channels: %i, img_shape: (%i, %i), '
          % ((n_imgs, n_channels) + img_shape)
          + 'n_filters: %i, filter_shape: (%i, %i), pad: %i'
          % ((n_filters,) + filter_shape + (pad,)))

    # Setup arrays
    padding = (pad, pad)
    strides = (1, 1)
    img_h, img_w = img_shape
    filter_h, filter_w = filter_shape
    convout_h = img_h + 2*pad - filter_h + 1
    convout_w = img_w + 2*pad - filter_w + 1

    imgs_bc01_shape = (n_imgs, n_channels, img_h, img_w)
    filters_bc01_shape = (n_filters, n_channels, filter_h, filter_w)

    imgs_bc01 = np.random.randn(n_imgs, n_channels, img_h, img_w)
    imgs_c01b = np.transpose(imgs_bc01, (1, 2, 3, 0))
    filters_fc01 = np.random.randn(n_filters, n_channels, filter_h, filter_w)
    filters_c01f = np.transpose(filters_fc01, (1, 2, 3, 0))
    convout_bc01 = np.random.randn(n_imgs, n_filters, convout_h, convout_w)
    convout_c01b = np.transpose(convout_bc01, (1, 2, 3, 0))

    imgs_bc01_t = theano.shared(imgs_bc01.astype(theano.config.floatX))
    imgs_c01b_t = theano.shared(imgs_c01b.astype(theano.config.floatX))
    filters_fc01_t = theano.shared(filters_fc01.astype(theano.config.floatX))
    filters_c01f_t = theano.shared(filters_c01f.astype(theano.config.floatX))
    convout_bc01_t = theano.shared(convout_bc01.astype(theano.config.floatX))
    convout_c01b_t = theano.shared(convout_c01b.astype(theano.config.floatX))
    imgs_bc01_ca = ca.array(imgs_bc01)
    filters_fc01_ca = ca.array(filters_fc01)
    convout_bc01_ca = ca.array(convout_bc01)

    # Forward propagation
    print('fprop')
    convout_cc_op = FilterActs(stride=1, partial_sum=4, pad=pad)
    convout_cc_expr = convout_cc_op(imgs_c01b_t, filters_c01f_t)
    convout_cc_fun = theano.function([], convout_cc_expr)
    convout_cc = convout_cc_fun()
    convout_cc = np.transpose(convout_cc, (3, 0, 1, 2))

    def convout_ca_fun():
        convout = ca.nnet.conv_bc01(imgs_bc01_ca, filters_fc01_ca, padding,
                                    strides)
        return convout
    convout_ca = np.array(convout_ca_fun())
    print('         correct: ' + str(allclose(convout_ca, convout_cc)))
    duration_cc = avg_running_time(convout_cc_fun)
    duration_ca = avg_running_time(convout_ca_fun)
    print('   avg. duration: cuda_convnet: %.4f  ca: %.4f'
          % (duration_cc, duration_ca))
    print('         speedup: %.2f' % (duration_cc/duration_ca))
    del convout_cc_op
    del convout_cc_expr
    del convout_cc_fun

#     Back propagation, imgs
    print('bprop_imgs')
    dimgs_cc_op = ImageActs(stride=1, partial_sum=1, pad=pad)
    dimgs_cc_expr = dimgs_cc_op(convout_c01b_t, filters_c01f_t)
    dimgs_cc_fun = theano.function([], dimgs_cc_expr)
    dimgs_cc = dimgs_cc_fun()
    dimgs_cc = np.transpose(dimgs_cc, (3, 0, 1, 2))

    def dimgs_ca_fun():
        return ca.nnet.conv_bc01_bprop_imgs(filters_fc01_ca, convout_bc01_ca,
                                            img_shape, padding, strides)
    dimgs_ca = np.array(dimgs_ca_fun())
    print('         correct: ' + str(allclose(dimgs_ca, dimgs_cc)))
    duration_cc = avg_running_time(dimgs_cc_fun)
    duration_ca = avg_running_time(dimgs_ca_fun)
    print('   avg. duration: cuda_convnet: %.4f  ca: %.4f'
          % (duration_cc, duration_ca))
    print('         speedup: %.2f' % (duration_cc/duration_ca))
    del dimgs_cc_op
    del dimgs_cc_expr
    del dimgs_cc_fun

    # Back propagation, filters
    dfilters_cc_op = WeightActs(stride=1, partial_sum=1, pad=pad)
    dfilters_cc_expr = dfilters_cc_op(imgs_c01b_t, convout_c01b_t,
                                      T.as_tensor_variable(filter_shape))
    dfilters_cc_fun = theano.function([], dfilters_cc_expr)
    dfilters_cc = dfilters_cc_fun()[0]
    dfilters_cc = np.transpose(dfilters_cc, (3, 0, 1, 2))

    def dfilters_ca_fun():
        return ca.nnet.conv_bc01_bprop_filters(imgs_bc01_ca, convout_bc01_ca,
                                               filter_shape, padding, strides)
    dfilters_ca = np.array(dfilters_ca_fun())

    print('bprop_filters')
#    print(np.array(dfilters_ca)[-1,-1,...] / dfilters_cc[-1,-1,...])
    print('         correct: ' + str(allclose(dfilters_ca, dfilters_cc)))
    duration_cc = avg_running_time(dfilters_cc_fun)
    duration_ca = avg_running_time(dfilters_ca_fun)
    print('   avg. duration: cuda_convnet: %.4f  ca: %.4f'
          % (duration_cc, duration_ca))
    print('         speedup: %.2f' % (duration_cc/duration_ca))


def run():
    np.random.seed(1)
    # Configurations are given in the form
    # (n_imgs, n_channels, img_shape, n_filters, filter_shape, padding)
    configurations = [
        # From the original paper
        # http://arxiv.org/abs/1312.5851
        (128, 3, (32, 32), 96, (11, 11), 0),
        (128, 96, (32, 32), 256, (7, 7), 0),
        (128, 256, (16, 16), 384, (5, 5), 0),
        (128, 384, (16, 16), 384, (5, 5), 0),
        (128, 384, (16, 16), 384, (3, 3), 0),
        # From Sander Dieleman
        # http://benanne.github.io/2014/05/12/fft-convolutions-in-theano.html
#        (64, 3, (96, 96), 128, (16, 16), 0),
#        (64, 128, (32, 32), 64, (8, 8), 0),
#        (128, 32, (54, 54), 64, (6, 6), 0),
#        (128, 128, (16, 16), 128, (8, 8), 0),
#        (128, 1024, (32, 32), 128, (4, 4), 0), # out of memory error
        # Exotic shapes and padding
        (5, 3, (5, 5), 16, (3, 3), 1),
        (64, 32, (32, 32), 32, (5, 5), 2),
        (64, 1, (17, 19), 32, (7, 7), 4),
        (64, 3, (9, 16), 32, (7, 7), 4),
#         Typical CNN layers for CIFAR-10
        (128, 3, (32, 32), 64, (5, 5), 2),
        (128, 64, (16, 16), 64, (5, 5), 2),
        (128, 64, (8, 8), 64, (5, 5), 2),
    ]

    for conf in configurations:
        benchmark(*conf)


if __name__ == '__main__':
    run()
