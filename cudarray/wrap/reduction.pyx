cimport numpy as np
cimport reduction
from .array_data cimport ArrayData, float_ptr, int_ptr, is_int, is_float


max_op = MAX_OP
mean_op = MEAN_OP
min_op = MIN_OP
sum_op = SUM_OP

argmax_op = ARGMAX_OP
argmin_op = ARGMIN_OP


def _reduce(ReduceOp op, ArrayData a, unsigned int n, ArrayData out):
    if is_float(a):
        reduction.reduce(op, float_ptr(a), n, float_ptr(out))
    elif is_int(a):
        reduction.reduce(op, int_ptr(a), n, int_ptr(out))
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _reduce_mat(ReduceOp op, ArrayData a, unsigned int m, unsigned int n,
                bool reduce_leading, ArrayData out):
    if is_float(a):
        reduction.reduce_mat(op, float_ptr(a), m, n, reduce_leading,
                             float_ptr(out))
    elif is_int(a):
        reduction.reduce_mat(op, int_ptr(a), m, n, reduce_leading,
                             int_ptr(out))
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))



def _reduce_to_int(ReduceToIntOp op, ArrayData a, unsigned int n,
                   ArrayData out):
    if is_float(a):
        reduction.reduce_to_int(op, float_ptr(a), n, int_ptr(out))
    elif is_int(a):
        reduction.reduce_to_int(op, int_ptr(a), n, int_ptr(out))
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))


def _reduce_mat_to_int(ReduceToIntOp op, ArrayData a, unsigned int m,
                       unsigned int n, bool reduce_leading, ArrayData out):
    if is_float(a):
        reduction.reduce_mat_to_int(op, float_ptr(a), m, n, reduce_leading,
                                    int_ptr(out))
    elif is_int(a):
        reduction.reduce_mat_to_int(op, int_ptr(a), m, n, reduce_leading,
                                    int_ptr(out))
    else:
        raise ValueError('type %s not implemented' % str(a.dtype))
