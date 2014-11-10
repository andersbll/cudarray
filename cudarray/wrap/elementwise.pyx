cimport numpy as np
cimport elementwise
from .array_data cimport (ArrayData, bool_ptr, float_ptr, int_ptr, is_int,
                          is_float)

btype_inner = BROADCAST_INNER
btype_leading = BROADCAST_LEADING
btype_outer = BROADCAST_OUTER
btype_trailing = BROADCAST_TRAILING

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


def _binary(BinaryOp op, ArrayData a, ArrayData b, unsigned int n,
            ArrayData c):
    if is_float(a) and is_float(b):
        elementwise.binary(op, float_ptr(a), float_ptr(b), n, float_ptr(c))
    elif is_float(a) and is_int(b):
        elementwise.binary(op, float_ptr(a), int_ptr(b), n, float_ptr(c))
    elif is_int(a) and is_float(b):
        elementwise.binary(op, int_ptr(a), float_ptr(b), n, float_ptr(c))
    elif is_int(a) and is_int(b):
        elementwise.binary(op, int_ptr(a), int_ptr(b), n, int_ptr(c))
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _binary_scalar(BinaryOp op, ArrayData a, alpha, unsigned int n,
                   ArrayData b):
    if is_float(a):
        elementwise.binary_scalar(op, float_ptr(a), <float>alpha, n,
                                  float_ptr(b))
    elif is_int(a) and isinstance(alpha,  float):
        elementwise.binary_scalar(op, int_ptr(a), <float>alpha, n,
                                  float_ptr(b))
    elif is_int(a) and isinstance(alpha,  int):
        elementwise.binary_scalar(op, int_ptr(a), <int>alpha, n, int_ptr(b))
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), type(alpha)))


def _binary_broadcast(BinaryOp op, BroadcastType btype, ArrayData a,
        ArrayData b, unsigned int k, unsigned int m, unsigned int n,
        ArrayData c):
    if is_float(a) and is_float(b):
        elementwise.binary_broadcast(op, btype, float_ptr(a), float_ptr(b), k,
                                     m, n, float_ptr(c))
    elif is_float(a) and is_int(b):
        elementwise.binary_broadcast(op, btype, float_ptr(a), int_ptr(b), k, m,
                                     n, float_ptr(c))
    elif is_int(a) and is_float(b):
        elementwise.binary_broadcast(op, btype, int_ptr(a), float_ptr(b), k, m,
                                     n, float_ptr(c))
    elif is_int(a) and is_int(b):
        elementwise.binary_broadcast(op, btype, int_ptr(a), int_ptr(b), k, m,
                                     n, int_ptr(c))
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _binary_cmp(BinaryCmpOp op, ArrayData a, ArrayData b, unsigned int n,
           ArrayData c):
    if is_float(a) and is_float(b):
        elementwise.binary_cmp[float, float](op, float_ptr(a), float_ptr(b), n,
                               bool_ptr(c))
    elif is_float(a) and is_int(b):
        elementwise.binary_cmp[float, int](op, float_ptr(a), int_ptr(b), n,
                               bool_ptr(c))
    elif is_int(a) and is_float(b):
        elementwise.binary_cmp[int, float](op, int_ptr(a), float_ptr(b), n,
                               bool_ptr(c))
    elif is_int(a) and is_int(b):
        elementwise.binary_cmp[int, int](op, int_ptr(a), int_ptr(b), n,
                               bool_ptr(c))
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _binary_cmp_scalar(BinaryCmpOp op, ArrayData a, alpha, unsigned int n,
                       ArrayData b):
    if is_float(a):
        elementwise.binary_cmp_scalar[float, float](op, float_ptr(a), alpha, n,
                                                    bool_ptr(b))
    elif is_int(a) and isinstance(alpha,  float):
        elementwise.binary_cmp_scalar[int, float](op, int_ptr(a), alpha, n,
                                                  bool_ptr(b))
    elif is_int(a) and isinstance(alpha,  int):
        elementwise.binary_cmp_scalar[int, int](op, int_ptr(a), alpha, n,
                                                bool_ptr(b))
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), type(alpha)))


def _binary_cmp_broadcast(BinaryCmpOp op, BroadcastType btype, ArrayData a,
    ArrayData b, unsigned int k, unsigned int m, unsigned int n, ArrayData c):
    if is_float(a) and is_float(b):
        elementwise.binary_cmp_broadcast[float, float](op, btype, float_ptr(a),
            float_ptr(b), k, m, n, bool_ptr(c))
    elif is_float(a) and is_int(b):
        elementwise.binary_cmp_broadcast[float, int](op, btype, float_ptr(a), 
            int_ptr(b), k, m, n, bool_ptr(c))
    elif is_int(a) and is_float(b):
        elementwise.binary_cmp_broadcast[int, float](op, btype, int_ptr(a),
            float_ptr(b), k, m, n, bool_ptr(c))
    elif is_int(a) and is_int(b):
        elementwise.binary_cmp_broadcast[int, int](op, btype, int_ptr(a),
            int_ptr(b), k, m, n, bool_ptr(c))
    else:
        raise ValueError('types (%s, %s) not implemented'
                         % (str(a.dtype), str(b.dtype)))


def _unary(UnaryOp op, ArrayData a, unsigned int n, ArrayData b):
    if is_float(a):
        elementwise.unary(op, float_ptr(a), n, float_ptr(b))
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _clip(ArrayData a, a_min, a_max, unsigned int n, ArrayData b):
    if is_float(a):
        elementwise.clip[float](float_ptr(a), a_min, a_max, n, float_ptr(b))
    elif is_int(a):
        elementwise.clip[int](int_ptr(a), a_min, a_max, n, int_ptr(b))
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))
