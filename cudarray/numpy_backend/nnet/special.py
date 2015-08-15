import numpy as np


def softmax(X):
    e = np.exp(X - np.amax(X, axis=1, keepdims=True))
    return e/np.sum(e, axis=1, keepdims=True)


def categorical_cross_entropy(y_pred, y_true, eps=1e-15):
    # Assumes one-hot encoding.
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # XXX: do we need to normalize?
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return loss


def one_hot_encode(labels, n_classes, out=None):
    out_shape = (labels.size, n_classes)
    if labels.dtype != np.dtype(int):
        raise ValueError('labels.dtype must be int')
    if out is None:
        out = np.empty(out_shape)
    else:
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    out.fill(0)
    if labels.size == 1:
        out[0, labels] = 1
    else:
        for c in range(n_classes):
            out[labels == c, c] = 1
    return out


def one_hot_decode(one_hot, out=None):
    out_shape = (one_hot.shape[0],)
    if out is None:
        out = np.empty(out_shape, dtype=np.dtype(int))
    else:
        if out.dtype != np.dtype(int):
            raise ValueError('out.dtype must be int')
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    result = np.argmax(one_hot, axis=1)
    np.copyto(out, result)
    return out
