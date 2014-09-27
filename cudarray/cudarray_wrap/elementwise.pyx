cimport numpy as np
from .array_data cimport ArrayData
cimport elementwise


add_op = ADD_OP
div_op = DIV_OP
max_op = MAX_B_OP
min_op = MIN_B_OP
mul_op = MUL_OP
pow_op = POW_OP
sub_op = SUB_OP

abs_op = ABS_OP
exp_op = EXP_OP
log_op = LOG_OP
neg_op = NEG_OP
relu_op = RELU_OP
relu_d_op = RELU_D_OP
sigmoid_op = SIGMOID_OP
sigmoid_d_op = SIGMOID_D_OP
sqrt_op = SQRT_OP
tanh_op = TANH_OP
tanh_d_op = TANH_D_OP


def _binary(BinaryOp op, ArrayData a, ArrayData b, unsigned int n,
           ArrayData c):
    if a.dtype == np.dtype('float32'):
        elementwise.binary[float](op, <const float *>a.dev_ptr,
            <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _binary_inplace(BinaryOp op, ArrayData a, ArrayData b, unsigned int n):
    if a.dtype == np.dtype('float32'):
        elementwise.binary_inplace[float](op, <float *>a.dev_ptr,
                                          <const float *>b.dev_ptr, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _binary_broadcast(BinaryOp op, ArrayData a, ArrayData b, unsigned int m,
                     unsigned int n, bool broadcast_to_leading, ArrayData c):
    if a.dtype == np.dtype('float32'):
        elementwise.binary_broadcast[float](op, <const float *>a.dev_ptr,
            <const float *>b.dev_ptr, m, n, broadcast_to_leading,
            <float *>c.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _binary_broadcast_inplace(BinaryOp op, ArrayData a, ArrayData b, unsigned int m,
                     unsigned int n, bool broadcast_to_leading):
    if a.dtype == np.dtype('float32'):
        elementwise.binary_broadcast_inplace[float](op, <float *>a.dev_ptr,
            <const float *>b.dev_ptr, m, n, broadcast_to_leading)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _binary_scalar(BinaryOp op, ArrayData a, alpha, unsigned int n, ArrayData b):
    if a.dtype == np.dtype('float32'):
        elementwise.binary_scalar[float](op, <const float *>a.dev_ptr,
            <float>alpha, n, <float *>b.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _binary_scalar_inplace(BinaryOp op, ArrayData a, alpha, unsigned int n):
    if a.dtype == np.dtype('float32'):
        elementwise.binary_scalar_inplace[float](op, <float *>a.dev_ptr,
                                                 <float>alpha, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _unary(UnaryOp op, ArrayData a, unsigned int n, ArrayData b):
    if a.dtype == np.dtype('float32'):
        elementwise.unary[float](op, <const float *>a.dev_ptr, n,
                                 <float *>b.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _unary_inplace(UnaryOp op, ArrayData a, unsigned int n):
    if a.dtype == np.dtype('float32'):
        elementwise.unary_inplace[float](op, <float *>a.dev_ptr, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _clip(ArrayData a, a_min, a_max, unsigned int n, ArrayData b):
    if a.dtype == np.dtype('float32'):
        elementwise.clip[float](<const float *>a.dev_ptr, <float> a_min,
                                <float> a_max, n, <float *>b.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _clip_inplace(ArrayData a, a_min, a_max, unsigned int n):
    if a.dtype == np.dtype('float32'):
        elementwise.clip_inplace[float](<float *>a.dev_ptr, <float> a_min,
                                        <float> a_max, n)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))

