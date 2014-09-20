#include "cudarray/common.hpp"
#include "cudarray/elementwise.hpp"


namespace cudarray {

#define ELEMENTWISE_OP_DEF(name, type) \
  template<> \
  void name<type>(const type *a, const type *b, int n, type *c) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, b, n, c); \
  }

#define ELEMENTWISE_OP_DEFS(name, operation) \
  template <typename T> \
  __global__ void kernel_##name(const T *a, const T *b, int n, T *c) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      c[idx] = a[idx] operation b[idx]; \
    } \
  } \
  ELEMENTWISE_OP_DEF(name, float)

ELEMENTWISE_OP_DEFS(add, +)
ELEMENTWISE_OP_DEFS(sub, -)
ELEMENTWISE_OP_DEFS(mul, *)
ELEMENTWISE_OP_DEFS(div, /)


#define ELEMENTWISE_BROADCAST_OP_DEF(name, type) \
  template<> \
  void name<type>(const type *a, const type *b, int m, int n,\
                  bool broadcast_to_leading, type *c) { \
    if (broadcast_to_leading) { \
      kernel_##name<true, type><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
          (a, b, m, n, c); \
    } else { \
      kernel_##name<false, type><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
          (a, b, m, n, c); \
    } \
  }

// abll: grid stride looping is not ideal for broadcasting 
#define ELEMENTWISE_BROADCAST_OP_DEFS(name, operation) \
  template <bool broadcast_to_leading, typename T> \
  __global__ void kernel_##name(const T *a, const T *b, int m, int n, T *c) { \
    unsigned int n_threads = n*m; \
    CUDA_GRID_STRIDE_LOOP(idx, n_threads) { \
      if (broadcast_to_leading) { \
        c[idx] = a[idx] operation b[idx % n]; \
      } else { \
        c[idx] = a[idx] operation b[idx / m]; \
      } \
    } \
  } \
  ELEMENTWISE_BROADCAST_OP_DEF(name, float)

ELEMENTWISE_BROADCAST_OP_DEFS(add_broadcast, +)
ELEMENTWISE_BROADCAST_OP_DEFS(mul_broadcast, *)


#define ELEMENTWISE_INPLACE_OP_DEF(name, type) \
  template<> \
  void name<type>(type *x, const type *y, int n) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(x, y, n); \
  }

#define ELEMENTWISE_INPLACE_OP_DEFS(name, operation) \
  template <typename T> \
  __global__ void kernel_##name(T *x, const T *y, int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      x[idx] operation y[idx]; \
    } \
  } \
  ELEMENTWISE_INPLACE_OP_DEF(name, float)

ELEMENTWISE_INPLACE_OP_DEFS(add_inplace, +=)
ELEMENTWISE_INPLACE_OP_DEFS(sub_inplace, -=)
ELEMENTWISE_INPLACE_OP_DEFS(mul_inplace, *=)
ELEMENTWISE_INPLACE_OP_DEFS(div_inplace, /=)


#define SCALAR_OP_DEF(name, type) \
  template<> \
  void name<type>(const type *x, type alpha, int n, type *y) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(x, alpha, n, y); \
  }

#define SCALAR_OP_DEFS(name, operation) \
  template <typename T> \
  __global__ void kernel_##name(const T *x, T alpha, int n, T *y) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      y[idx] = x[idx] operation alpha; \
    } \
  } \
  SCALAR_OP_DEF(name, float)

SCALAR_OP_DEFS(add_scalar, +)
SCALAR_OP_DEFS(sub_scalar, -)
SCALAR_OP_DEFS(mul_scalar, *)
SCALAR_OP_DEFS(div_scalar, /)


#define SCALAR_INPLACE_OP_DEF(name, type) \
  template<> \
  void name<type>(type *x, type alpha, int n) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(x, alpha, n); \
  }

#define SCALAR_INPLACE_OP_DEFS(name, operation) \
  template <typename T> \
  __global__ void kernel_##name(T *x, T alpha, int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      x[idx] operation alpha; \
    } \
  } \
  SCALAR_INPLACE_OP_DEF(name, float)

SCALAR_INPLACE_OP_DEFS(add_scalar_inplace, +=)
SCALAR_INPLACE_OP_DEFS(sub_scalar_inplace, -=)
SCALAR_INPLACE_OP_DEFS(mul_scalar_inplace, *=)
SCALAR_INPLACE_OP_DEFS(div_scalar_inplace, /=)



#define UNARY_OP_DEF(name, type) \
  template<> \
  void name<type>(const type *x, int n, type *y) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(x, n, y); \
  }

#define UNARY_OP_DEFS(name, operation) \
  template <typename T> \
  __global__ void kernel_##name(const T *x, int n, T *y) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      operation; \
    } \
  } \
  UNARY_OP_DEF(name, float)

UNARY_OP_DEFS(abs, y[idx] = fabsf(x[idx]))
UNARY_OP_DEFS(exp, y[idx] = expf(x[idx]))
UNARY_OP_DEFS(log, y[idx] = logf(x[idx]))
UNARY_OP_DEFS(sqrt, y[idx] = sqrtf(x[idx]))
UNARY_OP_DEFS(tanh, y[idx] = tanhf(x[idx]))
//UNARY_OP_DEFS(sigmoid, y[idx] = 1.0/(1.0 + expf(-x[idx])))


#define UNARY_ARG_OP_DEF(name, type) \
  template<> \
  void name<type>(const type *x, type arg, int n, type *y) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(x, arg, n, y); \
  }

#define UNARY_ARG_OP_DEFS(name, operation) \
  template <typename T> \
  __global__ void kernel_##name(const T *x, T arg, int n, T *y) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      operation; \
    } \
  } \
  UNARY_ARG_OP_DEF(name, float)

UNARY_ARG_OP_DEFS(pow, y[idx] = powf(x[idx], arg))
UNARY_ARG_OP_DEFS(max, y[idx] = fmaxf(x[idx], arg))
UNARY_ARG_OP_DEFS(min, y[idx] = fminf(x[idx], arg))

}
