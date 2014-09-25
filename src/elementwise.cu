#include "cudarray/common.hpp"
#include "cudarray/elementwise.hpp"


namespace cudarray {

#define BINARY_OP_WRAPPERS(name, type) \
  template<> \
  void name<type>(const type *a, const type *b, unsigned int n, type *c) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, b, n, c); \
  } \
  template<> \
  void name##_inplace<type>(type *a, const type *b, unsigned int n) { \
    kernel_##name##_inplace<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, b, n); \
  } \
  template<> \
  void name##_scalar<type>(const type *a, type alpha, unsigned int n, \
                           type *b) { \
    kernel_##name##_scalar<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
        (a, alpha, n, b); \
  } \
  template<> \
  void name##_scalar_inplace<type>(type *a, type alpha, unsigned int n) { \
    kernel_##name##_scalar_inplace<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
        (a, alpha, n); \
  }


#define BINARY_OP_DEF(name, op, op_inplace, op_scalar, op_scalar_inplace) \
  template <typename T> \
  __global__ void kernel_##name(const T *a, const T *b, unsigned int n, \
                                T *c) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      op; \
    } \
  } \
  template <typename T> \
  __global__ void kernel_##name##_inplace(T *a, const T *b, unsigned int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      op_inplace; \
    } \
  } \
  template <typename T> \
  __global__ void kernel_##name##_scalar(const T *a, T alpha, \
                                         unsigned int n, T *b) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      op_scalar; \
    } \
  } \
  template <typename T> \
  __global__ void kernel_##name##_scalar_inplace(T *a, T alpha, \
                                                 unsigned int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      op_scalar_inplace; \
    } \
  } \
  BINARY_OP_WRAPPERS(name, float)

BINARY_OP_DEF(add,
              c[idx] = a[idx] + b[idx],
              a[idx] += b[idx],
              b[idx] = a[idx] + alpha,
              a[idx] += alpha)
BINARY_OP_DEF(sub,
              c[idx] = a[idx] - b[idx],
              a[idx] -= b[idx],
              b[idx] = a[idx] - alpha,
              a[idx] -= alpha)
BINARY_OP_DEF(mul,
              c[idx] = a[idx] * b[idx],
              a[idx] *= b[idx],
              b[idx] = a[idx] * alpha,
              a[idx] *= alpha)
BINARY_OP_DEF(div,
              c[idx] = a[idx] / b[idx],
              a[idx] /= b[idx],
              b[idx] = a[idx] / alpha,
              a[idx] /= alpha)
BINARY_OP_DEF(max,
              c[idx] = fmaxf(a[idx], b[idx]),
              a[idx] = fmaxf(a[idx], b[idx]),
              b[idx] = fmaxf(a[idx], alpha),
              a[idx] = fmaxf(a[idx], alpha))
BINARY_OP_DEF(min,
              c[idx] = fminf(a[idx], b[idx]),
              a[idx] = fminf(a[idx], b[idx]),
              b[idx] = fminf(a[idx], alpha),
              a[idx] = fminf(a[idx], alpha))
BINARY_OP_DEF(pow,
              c[idx] = powf(a[idx], b[idx]),
              a[idx] = powf(a[idx], b[idx]),
              b[idx] = powf(a[idx], alpha),
              a[idx] = powf(a[idx], alpha))


#define BINARY_BROADCAST_OP_WRAPPER(name, type) \
  template<> \
  void name##_broadcast<type>(const type *a, const type *b, unsigned int m, \
                  unsigned int n, bool broadcast_to_leading, type *c) { \
    if (broadcast_to_leading) { \
      kernel_##name##_broadcast<true, type> \
          <<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
          (a, b, m, n, c); \
    } else { \
      kernel_##name##_broadcast<false, type> \
          <<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
          (a, b, m, n, c); \
    } \
  } \
  template<> \
  void name##_broadcast_inplace<type>(type *a, const type *b, unsigned int m, \
      unsigned int n, bool broadcast_to_leading) { \
    if (broadcast_to_leading) { \
      kernel_##name##_broadcast_inplace<true, type> \
          <<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
          (a, b, m, n); \
    } else { \
      kernel_##name##_broadcast_inplace<false, type> \
          <<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>> \
          (a, b, m, n); \
    } \
  }

// abll: grid stride looping might not ideal for broadcasting 
#define BINARY_BROADCAST_OP_DEF(name, op_leading, op_trailing, \
                                op_inplace_leading, op_inplace_trailing) \
  template <bool broadcast_to_leading, typename T> \
  __global__ void kernel_##name##_broadcast(const T *a, const T *b, \
      unsigned int m, unsigned int n, T *c) { \
    CUDA_GRID_STRIDE_LOOP(idx, n*m) { \
      if (broadcast_to_leading) { \
        op_leading; \
      } else { \
        op_trailing; \
      } \
    } \
  } \
  template <bool broadcast_to_leading, typename T> \
  __global__ void kernel_##name##_broadcast_inplace(T *a, const T *b, \
      unsigned int m, unsigned int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n*m) { \
      if (broadcast_to_leading) { \
        op_inplace_leading; \
      } else { \
        op_inplace_trailing; \
      } \
    } \
  } \
  BINARY_BROADCAST_OP_WRAPPER(name, float)

BINARY_BROADCAST_OP_DEF(add,
                        c[idx] = a[idx] + b[idx % n],
                        c[idx] = a[idx] + b[idx / m],
                        a[idx] += b[idx % n],
                        a[idx] += b[idx / m])
BINARY_BROADCAST_OP_DEF(sub,
                        c[idx] = a[idx] - b[idx % n],
                        c[idx] = a[idx] - b[idx / m],
                        a[idx] -= b[idx % n],
                        a[idx] -= b[idx / m])
BINARY_BROADCAST_OP_DEF(mul,
                        c[idx] = a[idx] * b[idx % n],
                        c[idx] = a[idx] * b[idx / m],
                        a[idx] *= b[idx % n],
                        a[idx] *= b[idx / m])
BINARY_BROADCAST_OP_DEF(div,
                        c[idx] = a[idx] / b[idx % n],
                        c[idx] = a[idx] / b[idx / m],
                        a[idx] /= b[idx % n],
                        a[idx] /= b[idx / m])
BINARY_BROADCAST_OP_DEF(max,
                        c[idx] = fmaxf(a[idx], b[idx % n]),
                        c[idx] = fmaxf(a[idx], b[idx / m]),
                        a[idx] = fmaxf(a[idx], b[idx % n]),
                        a[idx] = fmaxf(a[idx], b[idx / m]))
BINARY_BROADCAST_OP_DEF(min,
                        c[idx] = fminf(a[idx], b[idx % n]),
                        c[idx] = fminf(a[idx], b[idx / m]),
                        a[idx] = fminf(a[idx], b[idx % n]),
                        a[idx] = fminf(a[idx], b[idx / m]))
BINARY_BROADCAST_OP_DEF(pow,
                        c[idx] = powf(a[idx], b[idx % n]),
                        c[idx] = powf(a[idx], b[idx / m]),
                        a[idx] = powf(a[idx], b[idx % n]),
                        a[idx] = powf(a[idx], b[idx / m]))


#define UNARY_OP_WRAPPER(name, type) \
  template<> \
  void name<type>(const type *a, unsigned int n, type *b) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, n, b); \
  } \
  template<> \
  void name##_inplace<type>(type *a, unsigned int n) { \
    kernel_##name##_inplace<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, n); \
  }

#define UNARY_OP_DEF(name, op, op_inplace) \
  template <typename T> \
  __global__ void kernel_##name(const T *a, unsigned int n, T *b) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      op; \
    } \
  } \
  template <typename T> \
  __global__ void kernel_##name##_inplace(T *a, unsigned int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      op_inplace; \
    } \
  } \
  UNARY_OP_WRAPPER(name, float)

UNARY_OP_DEF(abs,
             b[idx] = fabsf(a[idx]),
             a[idx] = fabsf(a[idx]))
UNARY_OP_DEF(exp,
             b[idx] = expf(a[idx]),
             a[idx] = expf(a[idx]))
UNARY_OP_DEF(log,
             b[idx] = logf(a[idx]),
             a[idx] = logf(a[idx]))
UNARY_OP_DEF(relu,
             b[idx] = fmaxf(0.0, a[idx]),
             a[idx] = fmaxf(0.0, a[idx]))
UNARY_OP_DEF(relu_d,
             b[idx] = a[idx] > 0.0 ? 1 : 0,
             a[idx] = a[idx] > 0.0 ? 1 : 0)
UNARY_OP_DEF(sigmoid,
             b[idx] = 1.0/(1.0 + expf(-a[idx])),
             a[idx] = 1.0/(1.0 + expf(-a[idx])))
UNARY_OP_DEF(sqrt,
             b[idx] = sqrtf(a[idx]),
             a[idx] = sqrtf(a[idx]))
UNARY_OP_DEF(tanh,
             b[idx] = tanhf(a[idx]),
             a[idx] = tanhf(a[idx]))



#define UNARY_TMP_OP_WRAPPER(name, op_tmp_var, op, op_inplace, type) \
  __global__ void kernel_##name(const type *a, unsigned int n, type *b) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      type tmp = op_tmp_var; \
      op; \
    } \
  } \
  __global__ void kernel_##name##_inplace(type *a, unsigned int n) { \
    CUDA_GRID_STRIDE_LOOP(idx, n) { \
      type tmp = op_tmp_var; \
      op_inplace; \
    } \
  } \
  template<> \
  void name<type>(const type *a, unsigned int n, type *b) { \
    kernel_##name<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, n, b); \
  } \
  template<> \
  void name##_inplace<type>(type *a, unsigned int n) { \
    kernel_##name##_inplace<<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, n); \
  } \

#define UNARY_TMP_OP_DEF(name, op_tmp_var, op, op_inplace) \
  UNARY_TMP_OP_WRAPPER(name, op_tmp_var, op, op_inplace, float)

UNARY_TMP_OP_DEF(sigmoid_d,
             1.0/(1.0 + expf(-a[idx])),
             b[idx] = tmp*(1-tmp),
             a[idx] = tmp*(1-tmp))
UNARY_TMP_OP_DEF(tanh_d,
             expf(2*a[idx]),
             b[idx] = (tmp-1)/(tmp+1),
             a[idx] = (tmp-1)/(tmp+1))

}
