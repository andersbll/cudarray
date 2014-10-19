import numpy as np


def transpose(a):
    if a.ndim != 2:
        raise ValueError('transpose is implemented for 2D arrays only')
    a_trans = a.view()
    a_trans.shape = (a.shape[1], a.shape[0])
    a_trans.transposed = True
    return a_trans


def reshape(a, newshape):
    size = a.size
    if isinstance(newshape, int):
        newshape = (newshape,)
    newsize = np.prod(newshape)
    if size != newsize:
        if newsize < 0:
            # negative newsize means there is a -1 in newshape
            newshape = list(newshape)
            newshape[newshape.index(-1)] = -size // newsize
            newshape = tuple(newshape)
        else:
            raise ValueError('cannot reshape %s to %s' % (a.shape, newshape))
    a_reshaped = a.view()
    a_reshaped.shape = newshape
    return a_reshaped


bool_ = np.int32
int_ = np.int32
float_ = np.float32
