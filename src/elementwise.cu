#include "cudarray/common.hpp"
#include "cudarray/elementwise.hpp"


namespace cudarray {


#define BINARY_OP(name, operation, inplace_operation) \
  struct name { \
    template <typename Ta, typename Tb, typename Tc> \
    __device__ static void binary(volatile Ta a, volatile Tb b, \
                                  volatile Tc &c) { \
      operation; \
    } \
    template <typename Ta, typename Tb> \
    __device__ static void binary_inplace(volatile Ta &a, volatile Tb b) { \
      inplace_operation; \
    } \
  };

BINARY_OP(add_op, c = a + b, a += b)
BINARY_OP(div_op, c = a / b, a /= b)
BINARY_OP(max_op, c = fmaxf(a, b), a = fmaxf(a, b))
BINARY_OP(min_op, c = fminf(a, b), a = fminf(a, b))
BINARY_OP(mul_op, c = a * b, a *= b)
BINARY_OP(pow_op, c = powf(a, b), a = powf(a, b))
BINARY_OP(sub_op, c = a - b, a -= b)


template<typename Ta, typename Tb, typename Tc, typename Op>
__global__ void kernel_binary(const Ta *a, const Tb *b, unsigned int n,
                              Tc *c) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    Op::binary(a[idx], b[idx], c[idx]);
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op>
__global__ void kernel_binary_inplace(Ta *a, const Tb *b, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    Op::binary_inplace(a[idx], b[idx]);
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op>
void binary(const Ta *a, const Tb *b, unsigned int n, Tc *c) {
  if (c == (Tc *) a) {
    kernel_binary_inplace<Tc, Tb, Tc, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (c, b, n);
  } else if (c == (Tc *) b) {
    kernel_binary_inplace<Tc, Ta, Tc, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (c, a, n);

  } else {
    kernel_binary<Ta, Tb, Tc, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (a, b, n, c);
  }
}

template<typename Ta, typename Tb,  typename Tc>
void binary(BinaryOp op, const Ta *a, const Tb *b, unsigned int n, Tc *c) {
  switch (op) {
    case ADD_OP:
      binary<Ta, Tb, Tc, add_op>(a, b, n, c);
      break;
    case DIV_OP:
      binary<Ta, Tb, Tc, div_op>(a, b, n, c);
      break;
    case MAX_B_OP:
      binary<Ta, Tb, Tc, max_op>(a, b, n, c);
      break;
    case MIN_B_OP:
      binary<Ta, Tb, Tc, min_op>(a, b, n, c);
      break;
    case MUL_OP:
      binary<Ta, Tb, Tc, mul_op>(a, b, n, c);
      break;
    case POW_OP:
      binary<Ta, Tb, Tc, pow_op>(a, b, n, c);
      break;
    case SUB_OP:
      binary<Ta, Tb, Tc, sub_op>(a, b, n, c);
      break;
  }
}

template void binary<float, float, float>(
    BinaryOp op, const float *a, const float *b, unsigned int n, float *c);
template void binary<float, int, float>(
    BinaryOp op, const float *a, const int *b, unsigned int n, float *c);
template void binary<int, float, float>(
    BinaryOp op, const int *a, const float *b, unsigned int n, float *c);
template void binary<int, int, int>(
    BinaryOp op, const int *a, const int *b, unsigned int n, int *c);





template<typename Ta, typename Talpha, typename Tb, typename Op>
__global__ void kernel_binary_scalar(const Ta *a, Talpha alpha, unsigned int n,
                                     Tb *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    Op::binary(a[idx], alpha, b[idx]);
  }
}

template<typename Ta, typename Talpha, typename Op>
__global__ void kernel_binary_scalar_inplace(Ta *a, Talpha alpha,
                                             unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    Op::binary_inplace(a[idx], alpha);
  }
}

template<typename Ta, typename Talpha, typename Tb, typename Op>
void binary_scalar(const Ta *a, Talpha alpha, unsigned int n, Tb *b) {
  if (b == (Tb *)a) {
    kernel_binary_scalar_inplace<Tb, Talpha, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (b, alpha, n);
  } else {
    kernel_binary_scalar<Ta, Talpha, Tb, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (a, alpha, n, b);
  }

}

template<typename Ta, typename Talpha, typename Tb>
void binary_scalar(BinaryOp op, const Ta *a, Talpha alpha, unsigned int n,
                   Tb *b) {
  switch (op) {
    case ADD_OP:
      binary_scalar<Ta, Talpha, Tb, add_op>(a, alpha, n, b);
      break;
    case DIV_OP:
      binary_scalar<Ta, Talpha, Tb, div_op>(a, alpha, n, b);
      break;
    case MAX_B_OP:
      binary_scalar<Ta, Talpha, Tb, max_op>(a, alpha, n, b);
      break;
    case MIN_B_OP:
      binary_scalar<Ta, Talpha, Tb, min_op>(a, alpha, n, b);
      break;
    case MUL_OP:
      binary_scalar<Ta, Talpha, Tb, mul_op>(a, alpha, n, b);
      break;
    case POW_OP:
      binary_scalar<Ta, Talpha, Tb, pow_op>(a, alpha, n, b);
      break;
    case SUB_OP:
      binary_scalar<Ta, Talpha, Tb, sub_op>(a, alpha, n, b);
      break;
  }
}

template void binary_scalar<float, float, float>(
    BinaryOp op, const float *a, float alpha, unsigned int n, float *c);
template void binary_scalar<float, int, float>(
    BinaryOp op, const float *a, int alpha, unsigned int n, float *c);
template void binary_scalar<int, float, float>(
    BinaryOp op, const int *a, float alpha, unsigned int n, float *c);
template void binary_scalar<int, int, int>(
    BinaryOp op, const int *a, int alpha, unsigned int n, int *c);






template<typename Ta, typename Tb, typename Tc, typename Op, bool broadcast_leading>
__global__ void kernel_binary_broadcast(
    const Ta *a, const Tb *b, unsigned int m, unsigned int n, Tc *c) {
  CUDA_GRID_STRIDE_LOOP(idx, m*n) {
    if (broadcast_leading) {
      Op::binary(a[idx], b[idx % n], c[idx]);
    } else {
      Op::binary(a[idx], b[idx / m], c[idx]);
    }
  }
}

template<typename Ta, typename Tb, typename Op, bool broadcast_leading>
__global__ void kernel_binary_broadcast_inplace(
    Ta *a, const Tb *b, unsigned int m, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, m*n) {
    if (broadcast_leading) {
      Op::binary_inplace(a[idx], b[idx % n]);
    } else {
      Op::binary_inplace(a[idx], b[idx / m]);
    }
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op, bool broadcast_leading>
void binary_broadcast(const Ta *a, const Tb *b, unsigned int m,
                        unsigned int n, Tc *c) {
  if (c == (Tc *) a) {
    kernel_binary_broadcast_inplace<Ta, Tb, Op, broadcast_leading>
        <<<cuda_blocks(m*n), kNumBlockThreads>>>
        ((Ta *) c, b, m, n);
  } else if (c == (Tc *) b) {
    kernel_binary_broadcast_inplace<Tb, Ta, Op, broadcast_leading>
        <<<cuda_blocks(m*n), kNumBlockThreads>>>
        ((Tb *) b, a, m, n);

  } else {
    kernel_binary_broadcast<Ta, Tb, Tc, Op, broadcast_leading>
        <<<cuda_blocks(m*n), kNumBlockThreads>>>
        (a, b, m, n, c);
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op, bool broadcast_inner>
__global__ void kernel_binary_broadcast(const Ta *a, const Tb *b,
    unsigned int k, unsigned int m, unsigned int n, Tc *c) {
  CUDA_GRID_STRIDE_LOOP(idx, k*m*n) {
    if (broadcast_inner) {
      Op::binary(a[idx], b[idx / k / n], c[idx]);
    } else {
      Op::binary(a[idx], b[(idx / n) % m], c[idx]);
    }
  }
}

template<typename Ta, typename Tb, typename Op, bool broadcast_inner>
__global__ void kernel_binary_broadcast_inplace(Ta *a, const Tb *b,
    unsigned int k, unsigned int m, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, k*m*n) {
    if (broadcast_inner) {
      Op::binary_inplace(a[idx], b[idx / k / n]);
    } else {
      Op::binary_inplace(a[idx], b[(idx / n) % m]);
    }
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op, bool broadcast_inner>
void binary_broadcast(const Ta *a, const Tb *b, unsigned int k, unsigned int m,
                      unsigned int n, Tc *c) {
  if (c == (Tc *) a) {
    kernel_binary_broadcast_inplace<Ta, Tb, Op, broadcast_inner>
        <<<cuda_blocks(k*m*n), kNumBlockThreads>>>
        ((Ta *) c, b, k, m, n);
  } else if (c == (Tc *) b) {
    kernel_binary_broadcast_inplace<Tb, Ta, Op, broadcast_inner>
        <<<cuda_blocks(k*m*n), kNumBlockThreads>>>
        ((Tb *) b, a, k, m, n);

  } else {
    kernel_binary_broadcast<Ta, Tb, Tc, Op, broadcast_inner>
        <<<cuda_blocks(k*m*n), kNumBlockThreads>>>
        (a, b, k, m, n, c);
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op>
void binary_broadcast(BroadcastType btype, const Ta *a, const Tb *b,
    unsigned int k, unsigned int m, unsigned int n, Tc *c) {
  switch (btype) {
    case BROADCAST_INNER:
      binary_broadcast<Ta, Tb, Tc, Op, true>(a, b, k, m, n, c);
    case BROADCAST_LEADING:
      binary_broadcast<Ta, Tb, Tc, Op, true>(a, b, m, n, c);
      break;
    case BROADCAST_OUTER:
      binary_broadcast<Ta, Tb, Tc, Op, false>(a, b, k, m, n, c);
      break;
    case BROADCAST_TRAILING:
      binary_broadcast<Ta, Tb, Tc, Op, false>(a, b, m, n, c);
      break;
  }
}

template<typename Ta, typename Tb, typename Tc>
void binary_broadcast(BinaryOp op, BroadcastType btype, const Ta *a,
    const Tb *b, unsigned int k, unsigned int m, unsigned int n, Tc *c) {
  switch (op) {
    case ADD_OP:
      binary_broadcast<Ta, Tb, Tc, add_op>(btype, a, b, k, m, n, c);
      break;
    case DIV_OP:
      binary_broadcast<Ta, Tb, Tc, div_op>(btype, a, b, k, m, n, c);
      break;
    case MAX_B_OP:
      binary_broadcast<Ta, Tb, Tc, max_op>(btype, a, b, k, m, n, c);
      break;
    case MIN_B_OP:
      binary_broadcast<Ta, Tb, Tc, min_op>(btype, a, b, k, m, n, c);
      break;
    case MUL_OP:
      binary_broadcast<Ta, Tb, Tc, mul_op>(btype, a, b, k, m, n, c);
      break;
    case POW_OP:
      binary_broadcast<Ta, Tb, Tc, pow_op>(btype, a, b, k, m, n, c);
      break;
    case SUB_OP:
      binary_broadcast<Ta, Tb, Tc, sub_op>(btype, a, b, k, m, n, c);
      break;
  }
}

template void binary_broadcast<float, float, float>(
    BinaryOp op, BroadcastType btype, const float *a, const float *b,
    unsigned int k, unsigned int m, unsigned int n, float *c);
template void binary_broadcast<float, int, float>(
    BinaryOp op, BroadcastType btype, const float *a, const int *b,
    unsigned int k, unsigned int m, unsigned int n, float *c);
template void binary_broadcast<int, float, float>(
    BinaryOp op, BroadcastType btype, const int *a, const float *b,
    unsigned int k, unsigned int m, unsigned int n, float *c);
template void binary_broadcast<int, int, int>(
    BinaryOp op, BroadcastType btype, const int *a, const int *b,
    unsigned int k, unsigned int m,unsigned int n, int *c);



BINARY_OP(eq_op, c = a == b, a = a == b)
BINARY_OP(gt_op, c = a > b, a = a > b)
BINARY_OP(gt_eq_op, c = a >= b, a = a >= b)
BINARY_OP(lt_op, c = a < b, a = a < b)
BINARY_OP(lt_eq_op, c = a <= b, a = a <= b)
BINARY_OP(neq_op, c = a != b, a = a != b)


template<typename Ta, typename Tb>
void binary_cmp(BinaryCmpOp op, const Ta *a, const Tb *b, unsigned int n,
                bool_t *c) {
  switch (op) {
    case EQ_OP:
      binary<Ta, Tb, bool_t, eq_op>(a, b, n, c);
      break;
    case GT_OP:
      binary<Ta, Tb, bool_t, gt_op>(a, b, n, c);
      break;
    case GT_EQ_OP:
      binary<Ta, Tb, bool_t, gt_eq_op>(a, b, n, c);
      break;
    case LT_OP:
      binary<Ta, Tb, bool_t, lt_op>(a, b, n, c);
      break;
    case LT_EQ_OP:
      binary<Ta, Tb, bool_t, lt_eq_op>(a, b, n, c);
      break;
    case NEQ_OP:
      binary<Ta, Tb, bool_t, neq_op>(a, b, n, c);
      break;
  }
}

template void binary_cmp<float, float>(
    BinaryCmpOp op, const float *a, const float *b, unsigned int n, bool_t *c);
template void binary_cmp<float, int>(
    BinaryCmpOp op, const float *a, const int *b, unsigned int n, bool_t *c);
template void binary_cmp<int, float>(
    BinaryCmpOp op, const int *a, const float *b, unsigned int n, bool_t *c);
template void binary_cmp<int, int>(
    BinaryCmpOp op, const int *a, const int *b, unsigned int n, bool_t *c);



template<typename Ta, typename Talpha>
void binary_cmp_scalar(BinaryCmpOp op, const Ta *a, Talpha alpha,
                       unsigned int n, bool_t *b) {
  switch (op) {
    case EQ_OP:
      binary_scalar<Ta, Talpha, bool_t, eq_op>(a, alpha, n, b);
      break;
    case GT_OP:
      binary_scalar<Ta, Talpha, bool_t, gt_op>(a, alpha, n, b);
      break;
    case GT_EQ_OP:
      binary_scalar<Ta, Talpha, bool_t, gt_eq_op>(a, alpha, n, b);
      break;
    case LT_OP:
      binary_scalar<Ta, Talpha, bool_t, lt_op>(a, alpha, n, b);
      break;
    case LT_EQ_OP:
      binary_scalar<Ta, Talpha, bool_t, lt_eq_op>(a, alpha, n, b);
      break;
    case NEQ_OP:
      binary_scalar<Ta, Talpha, bool_t, neq_op>(a, alpha, n, b);
      break;
  }
}

template void binary_cmp_scalar<float, float>(
    BinaryCmpOp op, const float *a, float alpha, unsigned int n, bool_t *b);
template void binary_cmp_scalar<float, int>(
    BinaryCmpOp op, const float *a, int alpha, unsigned int n, bool_t *b);
template void binary_cmp_scalar<int, float>(
    BinaryCmpOp op, const int *a, float alpha, unsigned int n, bool_t *b);
template void binary_cmp_scalar<int, int>(
    BinaryCmpOp op, const int *a, int alpha, unsigned int n, bool_t *b);




template<typename Ta, typename Tb>
void binary_cmp_broadcast(BinaryCmpOp op, BroadcastType btype, const Ta *a,
    const Tb *b, unsigned int k, unsigned int m, unsigned int n, bool_t *c) {
  switch (op) {
    case EQ_OP:
      binary_broadcast<Ta, Tb, bool_t, eq_op>(btype, a, b, k, m, n, c);
      break;
    case GT_OP:
      binary_broadcast<Ta, Tb, bool_t, gt_op>(btype, a, b, k, m, n, c);
      break;
    case GT_EQ_OP:
      binary_broadcast<Ta, Tb, bool_t, gt_eq_op>(btype, a, b, k, m, n, c);
      break;
    case LT_OP:
      binary_broadcast<Ta, Tb, bool_t, lt_op>(btype, a, b, k, m, n, c);
      break;
    case LT_EQ_OP:
      binary_broadcast<Ta, Tb, bool_t, lt_eq_op>(btype, a, b, k, m, n, c);
      break;
    case NEQ_OP:
      binary_broadcast<Ta, Tb, bool_t, neq_op>(btype, a, b, k, m, n, c);
      break;
  }
}

template void binary_cmp_broadcast<float, float>(
    BinaryCmpOp op, BroadcastType btype, const float *a, const float *b,
    unsigned int k, unsigned int m, unsigned int n, bool_t *c);
template void binary_cmp_broadcast<float, int>(
    BinaryCmpOp op, BroadcastType btype, const float *a, const int *b,
    unsigned int k, unsigned int m, unsigned int n, bool_t *c);
template void binary_cmp_broadcast<int, float>(
    BinaryCmpOp op, BroadcastType btype, const int *a, const float *b,
    unsigned int k, unsigned int m, unsigned int n, bool_t *c);
template void binary_cmp_broadcast<int, int>(
    BinaryCmpOp op, BroadcastType btype, const int *a, const int *b,
    unsigned int k, unsigned int m, unsigned int n, bool_t *c);






#define UNARY_OP(name, operation, inplace_operation) \
  struct name { \
    template <typename T> \
    __device__ static void unary(volatile T a, volatile T &b) { \
      operation; \
    } \
    template <typename T> \
    __device__ static void unary_inplace(volatile T &a) { \
      inplace_operation; \
    } \
  };

UNARY_OP(abs_op, b = fabsf(a), a = fabsf(a))
UNARY_OP(cos_op, b = cosf(a), a = cosf(a))
UNARY_OP(exp_op, b = expf(a), a = expf(a))
UNARY_OP(log_op, b = logf(a), a = logf(a))
UNARY_OP(neg_op, b = -a, a = -a)
UNARY_OP(relu_op, b = fmaxf(0.0, a), a = fmaxf(0.0, a))
UNARY_OP(relu_d_op, b = a >= 0.0 ? 1.0 : 0.0, a = a >= 0.0 ? 1.0 : 0.0)
UNARY_OP(sigmoid_op, b = 1.0/(1.0 + expf(-a));, a = 1.0/(1.0 + expf(-a)))
UNARY_OP(sigmoid_d_op, T tmp = 1.0/(1.0 + expf(-a));  b = tmp*(1-tmp);,
         T tmp = 1.0/(1.0 + expf(-a));  a = tmp*(1-tmp))
UNARY_OP(sin_op, b = sinf(a), a = sinf(a))
UNARY_OP(sqrt_op, b = sqrtf(a), a = sqrtf(a))
UNARY_OP(tanh_op, b = tanhf(a), a = tanhf(a))
UNARY_OP(tanh_d_op, T tmp = tanhf(a); b = 1-tmp*tmp;,
         T tmp = tanhf(a); a = 1-tmp*tmp;)


template<typename T, typename Op>
__global__ void kernel_unary(const T *a, unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    Op::unary(a[idx], b[idx]);
  }
}

template<typename T, typename Op>
__global__ void kernel_unary_inplace(T *a, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    Op::unary_inplace(a[idx]);
  }
}

template<typename T, typename Op>
void unary(const T *a, unsigned int n, T *b) {
  if (a == b) {
    kernel_unary_inplace<T, Op><<<cuda_blocks(n), kNumBlockThreads>>>(b, n);
  } else {
    kernel_unary<T, Op><<<cuda_blocks(n), kNumBlockThreads>>>(a, n, b);
  }
}

template<typename T>
void unary(UnaryOp op, const T *a, unsigned int n, T *b) {
  switch (op) {
    case ABS_OP:
      unary<T, abs_op>(a, n, b);
      break;
    case COS_OP:
      unary<T, cos_op>(a, n, b);
      break;
    case EXP_OP:
      unary<T, exp_op>(a, n, b);
      break;
    case LOG_OP:
      unary<T, log_op>(a, n, b);
      break;
    case NEG_OP:
      unary<T, neg_op>(a, n, b);
      break;
    case RELU_OP:
      unary<T, relu_op>(a, n, b);
      break;
    case RELU_D_OP:
      unary<T, relu_d_op>(a, n, b);
      break;
    case SIGMOID_OP:
      unary<T, sigmoid_op>(a, n, b);
      break;
    case SIGMOID_D_OP:
      unary<T, sigmoid_d_op>(a, n, b);
      break;
    case SIN_OP:
      unary<T, sin_op>(a, n, b);
      break;
    case SQRT_OP:
      unary<T, sqrt_op>(a, n, b);
      break;
    case TANH_OP:
      unary<T, tanh_op>(a, n, b);
      break;
    case TANH_D_OP:
      unary<T, tanh_d_op>(a, n, b);
      break;
  }
}
template void unary<float>(UnaryOp op, const float *a, unsigned int n,
                           float *b);





template<typename T>
__global__ void kernel_clip(const T *a, T a_min, T a_max, unsigned int n,
                            T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = fminf(fmaxf(a[idx], a_min), a_max);
  }
}

template<typename T>
__global__ void kernel_clip_inplace(T *a, T a_min, T a_max, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = fminf(fmaxf(a[idx], a_min), a_max);
  }
}

template<typename T>
void clip(const T *a, T a_min, T a_max, unsigned int n, T *b) {
  if (a == b) {
    kernel_clip_inplace<T><<<cuda_blocks(n), kNumBlockThreads>>>
        (b, a_min, a_max, n);
  } else {
    kernel_clip<T><<<cuda_blocks(n), kNumBlockThreads>>>
        (a, a_min, a_max, n, b);
  }
}

template void clip<float>(const float *a, float a_min, float a_max,
                          unsigned int n, float *b);
template void clip<int>(const int *a, int a_min, int a_max, unsigned int n,
                        int *b);

}
