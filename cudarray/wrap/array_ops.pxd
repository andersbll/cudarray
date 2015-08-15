cdef extern from 'cudarray/array_ops.hpp' namespace 'cudarray':

    void transpose[T](const T *a, unsigned int n, unsigned int m, T *b)

    void as[Ta, Tb](const Ta *a, unsigned int n, Tb *b)

    void fill[T](T *a, unsigned int n, T alpha)

    void copy[T](const T *a, unsigned int n, T *b)

    void to_device[T](const T *a, unsigned int n, T *b)

    void to_host[T](const T *a, unsigned int n, T *b)
