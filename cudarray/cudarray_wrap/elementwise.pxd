from libcpp cimport bool

cdef extern from "cudarray/elementwise.hpp" namespace 'cudarray':
    void add[T](const T *a, const T *b, unsigned int n, T *c)
    void sub[T](const T *a, const T *b, unsigned int n, T *c)
    void mul[T](const T *a, const T *b, unsigned int n, T *c)
    void div[T](const T *a, const T *b, unsigned int n, T *c)
    void max[T](const T *a, const T *b, unsigned int n, T *c)
    void min[T](const T *a, const T *b, unsigned int n, T *c)
    void pow[T](const T *a, const T *b, unsigned int n, T *c)

    void add_inplace[T](const T *a, const T *b, unsigned int n, T *c)
    void sub_inplace[T](const T *a, const T *b, unsigned int n, T *c)
    void mul_inplace[T](const T *a, const T *b, unsigned int n, T *c)
    void div_inplace[T](const T *a, const T *b, unsigned int n, T *c)
    void max_inplace[T](const T *a, const T *b, unsigned int n, T *c)
    void min_inplace[T](const T *a, const T *b, unsigned int n, T *c)
    void pow_inplace[T](const T *a, const T *b, unsigned int n, T *c)

    void add_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)
    void sub_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)
    void mul_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)
    void div_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)
    void max_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)
    void min_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)
    void pow_broadcast[T](const T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading, T *c)

    void add_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)
    void sub_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)
    void mul_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)
    void div_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)
    void max_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)
    void min_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)
    void pow_broadcast_inplace[T](T *a, const T *b, unsigned int m,
        unsigned int n, bool broadcast_to_leading)

    void add_scalar[T](const T *a, const T b, unsigned int n, T *c)
    void sub_scalar[T](const T *a, const T b, unsigned int n, T *c)
    void mul_scalar[T](const T *a, const T b, unsigned int n, T *c)
    void div_scalar[T](const T *a, const T b, unsigned int n, T *c)
    void max_scalar[T](const T *a, const T b, unsigned int n, T *c)
    void min_scalar[T](const T *a, const T b, unsigned int n, T *c)
    void pow_scalar[T](const T *a, const T b, unsigned int n, T *c)

    void add_scalar_inplace[T](T *a, const T b, unsigned int n)
    void sub_scalar_inplace[T](T *a, const T b, unsigned int n)
    void mul_scalar_inplace[T](T *a, const T b, unsigned int n)
    void div_scalar_inplace[T](T *a, const T b, unsigned int n)
    void max_scalar_inplace[T](T *a, const T b, unsigned int n)
    void min_scalar_inplace[T](T *a, const T b, unsigned int n)
    void pow_scalar_inplace[T](T *a, const T b, unsigned int n)

    void abs[T](const T *a, unsigned int n, T *b)
    void exp[T](const T *a, unsigned int n, T *b)
    void log[T](const T *a, unsigned int n, T *b)
    void relu[T](const T *a, unsigned int n, T *b)
    void relu_d[T](const T *a, unsigned int n, T *b)
    void sigmoid[T](const T *a, unsigned int n, T *b)
    void sigmoid_d[T](const T *a, unsigned int n, T *b)
    void sqrt[T](const T *a, unsigned int n, T *b)
    void tanh[T](const T *a, unsigned int n, T *b)
    void tanh_d[T](const T *a, unsigned int n, T *b)

    void abs_inplace[T](T *a, unsigned int n)
    void exp_inplace[T](T *a, unsigned int n)
    void log_inplace[T](T *a, unsigned int n)
    void relu_inplace[T](const T *a, unsigned int n, T *b)
    void relu_d_inplace[T](const T *a, unsigned int n, T *b)
    void sigmoid_inplace[T](const T *a, unsigned int n, T *b)
    void sigmoid_d_inplace[T](const T *a, unsigned int n, T *b)
    void sqrt_inplace[T](T *a, unsigned int n)
    void tanh_inplace[T](T *a, unsigned int n)
    void tanh_d_inplace[T](const T *a, unsigned int n, T *b)
