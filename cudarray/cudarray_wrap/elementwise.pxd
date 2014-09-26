from libcpp cimport bool

cdef extern from 'cudarray/elementwise.hpp' namespace 'cudarray':
    enum BinaryOp:
        ADD_OP
        DIV_OP
        MAX_B_OP
        MIN_B_OP
        MUL_OP
        POW_OP
        SUB_OP

    void binary[T](BinaryOp op, const T *a, const T *b, unsigned int n, T *c);
    void binary_inplace[T](BinaryOp op, T *a, const T *b, unsigned int n);
    void binary_scalar[T](BinaryOp op, const T *a, T alpha, unsigned int n,
                          T *b);
    void binary_scalar_inplace[T](BinaryOp op, T *a, T alpha, unsigned int n);
    void binary_broadcast[T](BinaryOp op, const T *a, const T *b,
        unsigned int m, unsigned int n, bool broadcast_to_leading, T *c);
    void binary_broadcast_inplace[T](BinaryOp op, T *a, const T *b,
        unsigned int m, unsigned int n, bool broadcast_to_leading);

    enum UnaryOp:
        ABS_OP
        EXP_OP
        LOG_OP
        RELU_OP
        RELU_D_OP
        SIGMOID_OP
        SIGMOID_D_OP
        SQRT_OP
        TANH_OP
        TANH_D_OP

    void unary[T](UnaryOp op, const T *a, unsigned int n, T *b);
    void unary_inplace[T](UnaryOp op, T *a, unsigned int n);
