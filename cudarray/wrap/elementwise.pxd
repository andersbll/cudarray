from libcpp cimport bool

cdef extern from 'cudarray/common.hpp' namespace 'cudarray':
    ctypedef int bool_t;

cdef extern from 'cudarray/elementwise.hpp' namespace 'cudarray':
    enum BroadcastType:
        BROADCAST_INNER
        BROADCAST_LEADING
        BROADCAST_OUTER
        BROADCAST_TRAILING

    enum BinaryOp:
        ADD_OP
        DIV_OP
        MAX_B_OP
        MIN_B_OP
        MUL_OP
        POW_OP
        SUB_OP

    void binary[Ta, Tb, Tc](BinaryOp op, const Ta *a, const Tb *b,
                            unsigned int n, Tc *c)
    void binary_scalar[Ta, Talpha, Tb](BinaryOp op, const Ta *a, Talpha alpha,
                                       unsigned int n, Tb *b)
    void binary_scalar_[Talpha, Ta, Tb](BinaryOp op, Talpha alpha, const Ta *a,
                                        unsigned int n, Tb *b)
    void binary_broadcast[Ta, Tb, Tc](BinaryOp op, BroadcastType btype,
        const Ta *a, const Tb *b, unsigned int k, unsigned int m,
        unsigned int n, Tc *c)


    enum BinaryCmpOp:
        EQ_OP
        GT_OP
        GT_EQ_OP
        LT_OP
        LT_EQ_OP
        NEQ_OP

    void binary_cmp[Ta, Tb](BinaryCmpOp op, const Ta *a, const Tb *b,
                            unsigned int n, bool_t *c)
    void binary_cmp_scalar[T](BinaryCmpOp op, const T *a, T alpha,
        unsigned int n, bool_t *b)
    void binary_cmp_scalar_[T](BinaryCmpOp op, T alpha, const T *a,
        unsigned int n, bool_t *b)
    void binary_cmp_broadcast[Ta, Tb](BinaryCmpOp op, BroadcastType btype,
        const Ta *a, const Tb *b, unsigned int k, unsigned int m,
        unsigned int n, bool_t *c)


    enum UnaryOp:
        ABS_OP
        COS_OP
        EXP_OP
        LOG_OP
        LOG1P_OP
        NEG_OP
        RELU_OP
        RELU_D_OP
        SIGMOID_OP
        SIGMOID_D_OP
        SOFTPLUS_OP
        SOFTPLUS_D_OP
        SIN_OP
        SQRT_OP
        TANH_OP
        TANH_D_OP

    void unary[T](UnaryOp op, const T *a, unsigned int n, T *b)

    void clip[T](const T *a, T a_min, T a_max, unsigned int n, T *b)
