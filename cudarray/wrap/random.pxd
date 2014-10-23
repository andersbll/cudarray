cdef extern from 'cudarray/random.hpp' namespace 'cudarray':
    void seed(unsigned long long val);

    void random_normal[T](T *a, T mu, T sigma, unsigned int n);

    void random_uniform[T](T *a, T low, T high, unsigned int n);
