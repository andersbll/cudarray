import numpy as np
import cudarray as ca
from ..cudarray_wrap import nnet as wrap


def softmax(x):
    e = ca.exp(x - ca.amax(x, axis=1, keepdims=True))
    return e/ca.sum(e, axis=1, keepdims=True)


def categorical_cross_entropy(y_pred, y_true, eps=1e-15):
    # Assumes one-hot encoding.
    y_pred = ca.clip(y_pred, eps, 1 - eps)
    # XXX: do we need to normalize?
    y_pred /= ca.sum(y_pred, axis=1, keepdims=True)
    loss = -ca.sum(y_true * ca.log(y_pred), axis=1)
    return loss


def one_hot_encode(labels, n_classes, out=None):
    out_shape = (labels.size, n_classes)
    if labels.dtype != np.dtype('int32'):
        raise ValueError('labels.dtype must be int')
    if out is None:
        out = ca.empty(out_shape)
    else:
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    wrap._one_hot_encode(labels._data, n_classes, out_shape[0], out._data)
    return out


def one_hot_decode(one_hot, out=None):
    out_shape = (one_hot.shape[0],)
    if out is None:
        out = ca.empty(out_shape, dtype=np.dtype('int32'))
    else:
        if out.dtype != np.dtype('int32'):
            raise ValueError('out.dtype must be int')
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    ca.argmax(one_hot, axis=1, out=out)
    return out
