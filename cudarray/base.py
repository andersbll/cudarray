import numpy as np
import cudarray
from .wrap import array_ops


def transpose(a):
    if a.ndim != 2:
        raise ValueError('transpose is implemented for 2D arrays only')
    a_trans = a.view()
    a_trans.shape = (a.shape[1], a.shape[0])
    a_trans.transposed = True
    return a_trans


def reshape(a, newshape):
    a = ascontiguousarray(a)
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


def copyto(dst, src):
    if src.shape != dst.shape:
        raise ValueError('out.shape does not match result')
    if src.dtype != dst.dtype:
        raise ValueError('dtype mismatch')
    n = src.size
    if isinstance(src, np.ndarray):
        if isinstance(dst, np.ndarray):
            np.copyto(dst, src)
        else:
            dst = ascontiguousarray(dst)
            array_ops._to_device(src, n, dst._data)
    else:
        src = ascontiguousarray(src)
        if isinstance(dst, np.ndarray):
            array_ops._to_host(src._data, n, dst)
        else:
            dst = ascontiguousarray(dst)
            array_ops._copy(src._data, n, dst._data)


def ascontiguousarray(a):
    if not a.transposed:
        return a
    out = cudarray.empty_like(a)
    n, m = a.shape
    array_ops._transpose(a._data, m, n, out._data)
    return out


bool_ = np.int32
int_ = np.int32
float_ = np.float32
