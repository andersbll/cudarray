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
cos_op = COS_OP
exp_op = EXP_OP
log_op = LOG_OP
neg_op = NEG_OP
relu_op = RELU_OP
relu_d_op = RELU_D_OP
sigmoid_op = SIGMOID_OP
sigmoid_d_op = SIGMOID_D_OP
sin_op = SIN_OP
sqrt_op = SQRT_OP
tanh_op = TANH_OP
tanh_d_op = TANH_D_OP

eq_op = EQ_OP
gt_op = GT_OP
gt_eq_op = GT_EQ_OP
lt_op = LT_OP
lt_eq_op = LT_EQ_OP
neq_op = NEQ_OP


def isfloat(x):
    if isinstance(x, float):
        return True
    elif isinstance(x, ArrayData):
        return x.dtype == np.dtype('float32')
    else:
        return False


def isint(x):
    if isinstance(x, (int, long)):
        return True
    elif isinstance(x, ArrayData):
        return x.dtype == np.dtype('int32')
    else:
        return False


def _binary(BinaryOp op, ArrayData a, ArrayData b, unsigned int n,
           ArrayData c):
    if isfloat(a) and isfloat(b):
        elementwise.binary[float, float, float](op, <const float *>a.dev_ptr,
            <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
    elif isfloat(a) and isint(b):
        elementwise.binary[float, int, float](op, <const float *>a.dev_ptr,
            <const int *>b.dev_ptr, n, <float *>c.dev_ptr)
    elif isint(a) and isfloat(b):
        elementwise.binary[int, float, float](op, <const int *>a.dev_ptr,
            <const float *>b.dev_ptr, n, <float *>c.dev_ptr)
    elif isint(a) and isint(b):
        elementwise.binary[int, int, int](op, <const int *>a.dev_ptr,
            <const int *>b.dev_ptr, n, <int *>c.dev_ptr)
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _binary_scalar(BinaryOp op, ArrayData a, alpha, unsigned int n,
                   ArrayData b):
    if isfloat(a) and isfloat(alpha):
        elementwise.binary_scalar[float, float, float](op,
            <const float *>a.dev_ptr, <float>alpha, n, <float *>b.dev_ptr)
    elif isfloat(a) and isint(alpha):
        elementwise.binary_scalar[float, int, float](op,
            <const float *>a.dev_ptr, <int>alpha, n, <float *>b.dev_ptr)
    elif isint(a) and isfloat(alpha):
        elementwise.binary_scalar[int, float, float](op,
            <const int *>a.dev_ptr, <float>alpha, n, <float *>b.dev_ptr)
    elif isint(a) and isint(alpha):
        elementwise.binary_scalar[int, int, int](op, <const int *>a.dev_ptr,
            <int>alpha, n, <int *>b.dev_ptr)
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), type(alpha)))


def _binary_broadcast(BinaryOp op, ArrayData a, ArrayData b, unsigned int m,
                     unsigned int n, bool broadcast_to_leading, ArrayData c):
    if isfloat(a) and isfloat(b):
        elementwise.binary_broadcast[float, float, float](
            op, <const float *>a.dev_ptr, <const float *>b.dev_ptr, m, n,
            broadcast_to_leading, <float *>c.dev_ptr)
    elif isfloat(a) and isint(b):
        elementwise.binary_broadcast[float, int, float](
            op, <const float *>a.dev_ptr, <const int *>b.dev_ptr, m, n,
            broadcast_to_leading, <float *>c.dev_ptr)
    elif isint(a) and isfloat(b):
        elementwise.binary_broadcast[int, float, float](
            op, <const int *>a.dev_ptr, <const float *>b.dev_ptr, m, n,
            broadcast_to_leading, <float *>c.dev_ptr)
    elif isint(a) and isint(b):
        elementwise.binary_broadcast[int, int, int](
            op, <const int *>a.dev_ptr, <const int *>b.dev_ptr, m, n,
            broadcast_to_leading, <int *>c.dev_ptr)
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _binary_cmp(BinaryCmpOp op, ArrayData a, ArrayData b, unsigned int n,
           ArrayData c):
    if isfloat(a) and isfloat(b):
        elementwise.binary_cmp[float, float](op, <const float *>a.dev_ptr,
            <const float *>b.dev_ptr, n, <bool_t *>c.dev_ptr)
    elif isfloat(a) and isint(b):
        elementwise.binary_cmp[float, int](op, <const float *>a.dev_ptr,
            <const int *>b.dev_ptr, n, <bool_t *>c.dev_ptr)
    elif isint(a) and isfloat(b):
        elementwise.binary_cmp[float, int](op, <const float *>a.dev_ptr,
            <const int *>b.dev_ptr, n, <bool_t *>c.dev_ptr)
    elif isint(a) and isint(b):
        elementwise.binary_cmp[int, int](op, <const int *>a.dev_ptr,
            <const int *>b.dev_ptr, n, <bool_t *>c.dev_ptr)
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _binary_cmp_scalar(BinaryCmpOp op, ArrayData a, alpha, unsigned int n,
                       ArrayData b):
    if isfloat(a) and isfloat(alpha):
        elementwise.binary_cmp_scalar[float, float](op,
            <const float *>a.dev_ptr, <float>alpha, n, <bool_t *>b.dev_ptr)
    elif isfloat(a) and isint(alpha):
        elementwise.binary_cmp_scalar[float, int](op,
            <const float *>a.dev_ptr, <int>alpha, n, <bool_t *>b.dev_ptr)
    elif isint(a) and isfloat(alpha):
        elementwise.binary_cmp_scalar[int, float](op,
            <const int *>a.dev_ptr, <float>alpha, n, <bool_t *>b.dev_ptr)
    elif isint(a) and isint(alpha):
        elementwise.binary_cmp_scalar[int, int](op, <const int *>a.dev_ptr,
            <int>alpha, n, <bool_t *>b.dev_ptr)
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), type(alpha)))


def _binary_cmp_broadcast(BinaryCmpOp op, ArrayData a, ArrayData b,
    unsigned int m, unsigned int n, bool broadcast_to_leading, ArrayData c):
    if isfloat(a) and isfloat(b):
        elementwise.binary_cmp_broadcast[float, float](
            op, <const float *>a.dev_ptr, <const float *>b.dev_ptr, m, n,
            broadcast_to_leading, <bool_t *>c.dev_ptr)
    elif isfloat(a) and isint(b):
        elementwise.binary_cmp_broadcast[float, int](
            op, <const float *>a.dev_ptr, <const int *>b.dev_ptr, m, n,
            broadcast_to_leading, <bool_t *>c.dev_ptr)
    elif isint(a) and isfloat(b):
        elementwise.binary_cmp_broadcast[int, float](
            op, <const int *>a.dev_ptr, <const float *>b.dev_ptr, m, n,
            broadcast_to_leading, <bool_t *>c.dev_ptr)
    elif isint(a) and isint(b):
        elementwise.binary_cmp_broadcast[int, int](
            op, <const int *>a.dev_ptr, <const int *>b.dev_ptr, m, n,
            broadcast_to_leading, <bool_t *>c.dev_ptr)
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _unary(UnaryOp op, ArrayData a, unsigned int n, ArrayData b):
    if isfloat(a):
        elementwise.unary[float](op, <const float *>a.dev_ptr, n,
                                 <float *>b.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _clip(ArrayData a, a_min, a_max, unsigned int n, ArrayData b):
    if isfloat(a):
        elementwise.clip[float](<const float *>a.dev_ptr, <float> a_min,
                                <float> a_max, n, <float *>b.dev_ptr)
    elif isint(a):
        elementwise.clip[int](<const int *>a.dev_ptr, <int> a_min,
                                <int> a_max, n, <int *>b.dev_ptr)
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))
