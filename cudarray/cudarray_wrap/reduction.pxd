from libcpp cimport bool

cdef extern from "cudarray/reduction.hpp" namespace 'cudarray':
    void sum_mat[T](const T *a, unsigned int m, unsigned int n,
                    bool reduce_leading, T *b);
