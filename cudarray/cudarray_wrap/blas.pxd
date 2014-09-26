cdef extern from "cudarray/blas.hpp" namespace 'cudarray':
    enum TransposeOp:
        OP_TRANS
        OP_NO_TRANS

    T dot[T](const T *a, const T *b, unsigned int n);

    void gemv[T](const T *A, const T *b, TransposeOp trans, unsigned int m,
                 unsigned int n, T alpha, T beta, T *c);

    void gemm[T](const T *A, const T *B, TransposeOp transA,
                 TransposeOp transB, unsigned int m, unsigned int n,
                 unsigned int k, T alpha, T beta, T *C);
