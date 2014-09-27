import cudarray as ca


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
