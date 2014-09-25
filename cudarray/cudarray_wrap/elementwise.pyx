cimport numpy as np
from .array_data cimport ArrayData
cimport elementwise


class BinaryOp(object):
    add = 0
    sub = 1
    mul = 2
    div = 3
    max = 4
    min = 5
    pow = 6

class UnaryOp(object):
    abs = 0
    exp = 1
    log = 2
    relu = 3
    relu_d = 4
    sigmoid = 5
    sigmoid_d = 6
    sqrt = 7
    tanh = 8
    tanh_d = 9


def binary(int op, ArrayData a, ArrayData b, unsigned int n, ArrayData c):
    if a.dtype == np.dtype('float32'):
        if op == BinaryOp.add:
            elementwise.add[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        elif op == BinaryOp.sub:
            elementwise.sub[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        elif op == BinaryOp.mul:
            elementwise.mul[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        elif op == BinaryOp.div:
            elementwise.div[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        elif op == BinaryOp.max:
            elementwise.max[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        elif op == BinaryOp.min:
            elementwise.min[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        elif op == BinaryOp.pow:
            elementwise.pow[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
        else:
            raise ValueError('invalid op %i specified' % op)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def binary_inplace(int op, ArrayData a, ArrayData b, unsigned int n):
    if a.dtype == np.dtype('float32'):
        if op == BinaryOp.add:
            elementwise.add_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        elif op == BinaryOp.sub:
            elementwise.sub_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        elif op == BinaryOp.mul:
            elementwise.mul_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        elif op == BinaryOp.div:
            elementwise.div_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        elif op == BinaryOp.max:
            elementwise.max_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        elif op == BinaryOp.min:
            elementwise.min_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        elif op == BinaryOp.pow:
            elementwise.pow_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, n)
        else:
            raise ValueError('invalid op %i specified' % op)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def binary_broadcast(int op, ArrayData a, ArrayData b, unsigned int m,
                     unsigned int n, bool broadcast_to_leading, ArrayData c):
    if a.dtype == np.dtype('float32'):
        if op == BinaryOp.add:
            elementwise.add_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        elif op == BinaryOp.sub:
            elementwise.sub_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        elif op == BinaryOp.mul:
            elementwise.mul_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        elif op == BinaryOp.div:
            elementwise.div_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        elif op == BinaryOp.max:
            elementwise.max_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        elif op == BinaryOp.min:
            elementwise.min_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        elif op == BinaryOp.pow:
            elementwise.pow_broadcast[float](<const float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading,
                <float *>c.dev_ptr)
        else:
            raise ValueError('invalid op %i specified' % op)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def binary_broadcast_inplace(int op, ArrayData a, ArrayData b, unsigned int m,
                     unsigned int n, bool broadcast_to_leading):
    if a.dtype == np.dtype('float32'):
        pass
        if op == BinaryOp.add:
            elementwise.add_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        elif op == BinaryOp.sub:
            elementwise.sub_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        elif op == BinaryOp.mul:
            elementwise.mul_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        elif op == BinaryOp.div:
            elementwise.div_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        elif op == BinaryOp.max:
            elementwise.max_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        elif op == BinaryOp.min:
            elementwise.min_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        elif op == BinaryOp.pow:
            elementwise.pow_broadcast_inplace[float](<float *>a.dev_ptr,
                <const float *>b.dev_ptr, m, n, broadcast_to_leading)
        else:
            raise ValueError('invalid op %i specified' % op)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def scalar(int op, ArrayData a, alpha, unsigned int n, ArrayData b):
    if a.dtype == np.dtype('float32'):
        if op == BinaryOp.add:
            elementwise.add_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        elif op == BinaryOp.sub:
            elementwise.sub_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        elif op == BinaryOp.mul:
            elementwise.mul_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        elif op == BinaryOp.div:
            elementwise.div_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        elif op == BinaryOp.max:
            elementwise.max_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        elif op == BinaryOp.min:
            elementwise.min_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        elif op == BinaryOp.pow:
            elementwise.pow_scalar[float](<const float *>a.dev_ptr,
                <float>alpha, n, <float *>b.dev_ptr)
        else:
            raise ValueError('invalid op %i specified' % op)


def scalar_inplace(int op, ArrayData a, alpha, unsigned int n):
    if a.dtype == np.dtype('float32'):
        if op == BinaryOp.add:
            elementwise.add_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        elif op == BinaryOp.sub:
            elementwise.sub_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        elif op == BinaryOp.mul:
            elementwise.mul_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        elif op == BinaryOp.div:
            elementwise.div_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        elif op == BinaryOp.max:
            elementwise.max_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        elif op == BinaryOp.min:
            elementwise.min_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        elif op == BinaryOp.pow:
            elementwise.pow_scalar_inplace[float](<float *>a.dev_ptr,
                <float>alpha, n)
        else:
            raise ValueError('invalid op %i specified' % op)


def unary(int op, ArrayData a, unsigned int n, ArrayData b):
    if a.dtype == np.dtype('float32'):
        if op == UnaryOp.abs:
            elementwise.abs[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.exp:
            elementwise.exp[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.log:
            elementwise.log[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.relu:
            elementwise.relu[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.relu_d:
            elementwise.relu_d[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.sigmoid:
            elementwise.sigmoid[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.sigmoid_d:
            elementwise.sigmoid_d[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.sqrt:
            elementwise.sqrt[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.tanh:
            elementwise.tanh[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        if op == UnaryOp.tanh_d:
            elementwise.tanh_d[float](<const float *>a.dev_ptr, n,
                <float *>b.dev_ptr)
        else:
            raise ValueError('invalid op %i specified' % op)


def unary_inplace(int op, ArrayData a, unsigned int n, ArrayData b):
    if a.dtype == np.dtype('float32'):
        if op == UnaryOp.abs:
            elementwise.abs_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.exp:
            elementwise.exp_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.log:
            elementwise.log_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.relu:
            elementwise.relu_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.relu_d:
            elementwise.relu_d_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.sigmoid:
            elementwise.sigmoid_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.sigmoid_d:
            elementwise.sigmoid_d_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.sqrt:
            elementwise.sqrt_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.tanh:
            elementwise.tanh_inplace[float](<float *>a.dev_ptr, n)
        if op == UnaryOp.tanh_d:
            elementwise.tanh_d_inplace[float](<float *>a.dev_ptr, n)
        else:
            raise ValueError('invalid op %i specified' % op)
