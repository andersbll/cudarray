cimport numpy as np
cimport array_ops
from .array_data cimport (ArrayData, bool_ptr, float_ptr, int_ptr, is_int,
                          is_float)


def _concatenate(ArrayData a, ArrayData b, unsigned int axis, unsigned int d0,
                 unsigned int d1, unsigned int d2, unsigned int da,
                 unsigned int db, ArrayData c):
    if is_float(a):
        array_ops.concatenate(float_ptr(a), float_ptr(b), axis, d0, d1, d2, da,
                              db, float_ptr(c))
    elif is_int(a):
        array_ops.concatenate(int_ptr(a), int_ptr(b), axis, d0, d1, d2, da, db,
                              int_ptr(c))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _split(ArrayData c, unsigned int axis, unsigned int d0, unsigned int d1,
           unsigned int d2, unsigned int da, unsigned int db, ArrayData a,
           ArrayData b):
    if is_float(a):
        array_ops.split(float_ptr(c), axis, d0, d1, d2, da, db, float_ptr(a),
                        float_ptr(b))
    elif is_int(a):
        array_ops.split(int_ptr(c), axis, d0, d1, d2, da, db, int_ptr(a),
                        int_ptr(b))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _transpose(ArrayData a, unsigned int m, unsigned int n, ArrayData out):
    if is_float(a):
        array_ops.transpose(float_ptr(a), m, n, float_ptr(out))
    elif is_int(a):
        array_ops.transpose(int_ptr(a), m, n, int_ptr(out))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _asfloat(ArrayData a, unsigned int n, ArrayData out):
    if is_int(a):
        array_ops.as[int, float](int_ptr(a), n, float_ptr(out))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _asint(ArrayData a, unsigned int n, ArrayData out):
    if is_float(a):
        array_ops.as[float, int](float_ptr(a), n, int_ptr(out))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _fill(ArrayData a, unsigned int n, alpha):
    if is_int(a):
        array_ops.fill(int_ptr(a), n, <int>alpha)
    elif is_float(a):
        array_ops.fill(float_ptr(a), n, <float>alpha)
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _copy(ArrayData a, unsigned int n, ArrayData out):
    if is_int(a):
        array_ops.copy(int_ptr(a), n, int_ptr(out))
    elif is_float(a):
        array_ops.copy(float_ptr(a), n, float_ptr(out))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _to_device(np.ndarray a, unsigned int n, ArrayData out):
    if is_int(out):
        array_ops.to_device(<int *>np.PyArray_DATA(a), n, int_ptr(out))
    elif is_float(out):
        array_ops.to_device(<float *>np.PyArray_DATA(a), n, float_ptr(out))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))


def _to_host(ArrayData a, unsigned int n, np.ndarray out):
    if is_int(a):
        array_ops.to_host(int_ptr(a), n, <int *>np.PyArray_DATA(out))
    elif is_float(a):
        array_ops.to_host(float_ptr(a), n, <float *>np.PyArray_DATA(out))
    else:
        raise ValueError('type (%s) not implemented' % str(a.dtype))
