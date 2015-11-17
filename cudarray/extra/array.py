import cudarray as ca
from ..wrap import array_ops
from ..helpers import prod


def concatenate(a, b, axis=0, out=None):
    ndim = a.ndim
    a_shp = a.shape
    b_shp = b.shape

    d_concat = a_shp[axis] + b_shp[axis]
    out_shp = a_shp[:axis] + (d_concat,) + a_shp[axis+1:]
    if out is None:
        out = ca.empty(out_shp, dtype=a.dtype)
    else:
        if out.shape != out_shp:
            raise ValueError('shape mismatch')

    da = a_shp[axis]
    db = b_shp[axis]
    if ndim < 3:
        a_shp = a_shp + (1,)*(3-ndim)
        b_shp = b_shp + (1,)*(3-ndim)
    elif ndim > 3:
        if axis == 0:
            a_shp = a_shp[axis], prod(a_shp[1:]), 1
            b_shp = b_shp[axis], prod(b_shp[1:]), 1
        elif axis + 1 == ndim:
            a_shp = 1, prod(a_shp[:axis]), a_shp[axis]
            b_shp = 1, prod(b_shp[:axis]), b_shp[axis]
            axis = 2
        else:
            a_shp = prod(a_shp[:axis]), a_shp[axis], prod(a_shp[axis+1:])
            b_shp = prod(b_shp[:axis]), b_shp[axis], prod(b_shp[axis+1:])
            axis = 1
    d0, d1, d2 = a_shp[:axis] + (d_concat,) + a_shp[axis+1:]
    array_ops._concatenate(a._data, b._data, axis, d0, d1, d2, da, db,
                           out._data)
    return out


def split(arr, a_size, axis=0, out_a=None, out_b=None):
    shp = arr.shape
    ndim = arr.ndim
    da = a_size
    db = shp[axis]-a_size

    out_a_shp = shp[:axis] + (da,) + shp[axis+1:]
    out_b_shp = shp[:axis] + (db,) + shp[axis+1:]
    if out_a is None:
        out_a = ca.empty(out_a_shp, dtype=arr.dtype)
    else:
        if out_a.shape != out_a_shp:
            raise ValueError('shape mismatch')
    if out_b is None:
        out_b = ca.empty(out_b_shp, dtype=arr.dtype)
    else:
        if out_b.shape != out_b_shp:
            raise ValueError('shape mismatch')

    if ndim < 3:
        shp = shp + (1,)*(3-ndim)
    elif ndim > 3:
        if axis == 0:
            shp = shp[axis], prod(shp[1:]), 1
        elif axis + 1 == ndim:
            shp = 1, prod(shp[:axis]), shp[axis]
            axis = 2
        else:
            shp = prod(shp[:axis]), shp[axis], prod(shp[axis+1:])
            axis = 1

    d0, d1, d2 = shp
    array_ops._split(arr._data, axis, d0, d1, d2, da, db, out_a._data,
                     out_b._data)
    return out_a, out_b
