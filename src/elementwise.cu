#include "cudarray/common.hpp"
#include "cudarray/elementwise.hpp"


namespace cudarray {


#define BINARY_OP(name, operation) \
template <typename Ta, typename Tb, typename Tc> \
struct name { \
  __device__ Tc operator()(const Ta a, const Tb b) { \
    return operation; \
  } \
};

BINARY_OP(AddOp, a + b)
BINARY_OP(DivOp, a / b)
BINARY_OP(MaxOp, fmaxf(a, b))
BINARY_OP(MinOp, fminf(a, b))
BINARY_OP(MulOp, a * b)
BINARY_OP(PowOp, powf(a, b))
BINARY_OP(SubOp, a - b)



template<typename Ta, typename Tb, typename Tc, typename Op>
__global__ void kernel_binary(const Ta *a, const Tb *b, unsigned int n,
                              Tc *c) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    c[idx] = op(a[idx], b[idx]);
  }
}

template<typename Ta, typename Tb, typename Op>
__global__ void kernel_binary_inplace_a(Ta *a, const Tb *b, unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = op(a[idx], b[idx]);
  }
}

template<typename Tb, typename Ta, typename Op>
__global__ void kernel_binary_inplace_b(Tb *b, const Ta *a, unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = op(a[idx], b[idx]);
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op>
void binary(const Ta *a, const Tb *b, unsigned int n, Tc *c) {
  if (c == (Tc *) a) {
    kernel_binary_inplace_a<Tc, Tb, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (c, b, n);
  } else if (c == (Tc *) b) {
    kernel_binary_inplace_b<Tc, Ta, Op>
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
      binary<Ta, Tb, Tc, AddOp<Ta, Tb, Tc> >(a, b, n, c);
      break;
    case DIV_OP:
      binary<Ta, Tb, Tc, DivOp<Ta, Tb, Tc> >(a, b, n, c);
      break;
    case MAX_B_OP:
      binary<Ta, Tb, Tc, MaxOp<Ta, Tb, Tc> >(a, b, n, c);
      break;
    case MIN_B_OP:
      binary<Ta, Tb, Tc, MinOp<Ta, Tb, Tc> >(a, b, n, c);
      break;
    case MUL_OP:
      binary<Ta, Tb, Tc, MulOp<Ta, Tb, Tc> >(a, b, n, c);
      break;
    case POW_OP:
      binary<Ta, Tb, Tc, PowOp<Ta, Tb, Tc> >(a, b, n, c);
      break;
    case SUB_OP:
      binary<Ta, Tb, Tc, SubOp<Ta, Tb, Tc> >(a, b, n, c);
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
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = op(a[idx], alpha);
  }
}

template<typename Ta, typename Talpha, typename Op>
__global__ void kernel_binary_scalar_inplace(Ta *a, Talpha alpha,
                                             unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = op(a[idx], alpha);
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
      binary_scalar<Ta, Talpha, Tb, AddOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      break;
    case DIV_OP:
      binary_scalar<Ta, Talpha, Tb, DivOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      break;
    case MAX_B_OP:
      binary_scalar<Ta, Talpha, Tb, MaxOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      break;
    case MIN_B_OP:
      binary_scalar<Ta, Talpha, Tb, MinOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      break;
    case MUL_OP:
      binary_scalar<Ta, Talpha, Tb, MulOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      break;
    case POW_OP:
      if (alpha == static_cast<Talpha>(2)) {
        binary<Ta, Ta, Tb, MulOp<Ta, Talpha, Tb> >(a, a, n, b);
      } else if (alpha == static_cast<Talpha>(1)) {
        binary_scalar<Ta, Ta, Tb, MulOp<Ta, Talpha, Tb> >(a, 1, n, b);
      } else {
        binary_scalar<Ta, Talpha, Tb, PowOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      }
      break;
    case SUB_OP:
      binary_scalar<Ta, Talpha, Tb, SubOp<Ta, Talpha, Tb> >(a, alpha, n, b);
      break;
  }
}

template void binary_scalar<float, float, float>(
    BinaryOp op, const float *a, float alpha, unsigned int n, float *c);
template void binary_scalar<int, float, float>(
    BinaryOp op, const int *a, float alpha, unsigned int n, float *c);
template void binary_scalar<int, int, int>(
    BinaryOp op, const int *a, int alpha, unsigned int n, int *c);



template<typename Talpha, typename Ta, typename Tb, typename Op>
__global__ void kernel_binary_scalar(Talpha alpha, const Ta *a, unsigned int n,
                                     Tb *b) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = op(alpha, a[idx]);
  }
}

template<typename Talpha, typename Ta, typename Op>
__global__ void kernel_binary_scalar_inplace(Talpha alpha, Ta *a,
                                             unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = op(alpha, a[idx]);
  }
}

template<typename Talpha, typename Ta, typename Tb, typename Op>
void binary_scalar(Talpha alpha, const Ta *a, unsigned int n, Tb *b) {
  if (b == (Tb *)a) {
    kernel_binary_scalar_inplace<Talpha, Tb, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (alpha, b, n);
  } else {
    kernel_binary_scalar<Talpha, Ta, Tb, Op>
        <<<cuda_blocks(n), kNumBlockThreads>>>
        (alpha, a, n, b);
  }

}

template<typename Talpha, typename Ta, typename Tb>
void binary_scalar_(BinaryOp op, Talpha alpha, const Ta *a, unsigned int n,
                   Tb *b) {
  switch (op) {
    case ADD_OP:
      binary_scalar<Talpha, Ta, Tb, AddOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      break;
    case DIV_OP:
      binary_scalar<Talpha, Ta, Tb, DivOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      break;
    case MAX_B_OP:
      binary_scalar<Talpha, Ta, Tb, MaxOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      break;
    case MIN_B_OP:
      binary_scalar<Talpha, Ta, Tb, MinOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      break;
    case MUL_OP:
      binary_scalar<Talpha, Ta, Tb, MulOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      break;
    case POW_OP:
      if (alpha == static_cast<Talpha>(2)) {
        binary<Ta, Ta, Tb, MulOp<Talpha, Ta, Tb> >(a, a, n, b);
      } else if (alpha == static_cast<Talpha>(1)) {
        binary_scalar<Ta, Ta, Tb, MulOp<Talpha, Ta, Tb> >(1, a, n, b);
      } else {
        binary_scalar<Talpha, Ta, Tb, PowOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      }
      break;
    case SUB_OP:
      binary_scalar<Talpha, Ta, Tb, SubOp<Talpha, Ta, Tb> >(alpha, a, n, b);
      break;
  }
}

template void binary_scalar_<float, float, float>(
    BinaryOp op, float alpha, const float *a, unsigned int n, float *c);
template void binary_scalar_<float, int, float>(
    BinaryOp op, float alpha, const int *a, unsigned int n, float *c);
template void binary_scalar_<int, int, int>(
    BinaryOp op, int alpha, const int *a, unsigned int n, int *c);





template<typename Ta, typename Tb, typename Tc, typename Op, bool broadcast_leading>
__global__ void kernel_binary_broadcast(
    const Ta *a, const Tb *b, unsigned int m, unsigned int n, Tc *c) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, m*n) {
    if (broadcast_leading) {
      c[idx] = op(a[idx], b[idx % n]);
    } else {
      c[idx] = op(a[idx], b[idx / m]);
    }
  }
}

template<typename Ta, typename Tb, typename Op, bool broadcast_leading>
__global__ void kernel_binary_broadcast_inplace(
    Ta *a, const Tb *b, unsigned int m, unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, m*n) {
    if (broadcast_leading) {
      a[idx] = op(a[idx], b[idx % n]);
    } else {
      a[idx] = op(a[idx], b[idx / m]);
    }
  }
}

template<typename Ta, typename Tb, typename Tc, typename Op,
         bool broadcast_leading>
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

template<typename Ta, typename Tb, typename Tc, typename Op,
         bool broadcast_inner>
__global__ void kernel_binary_broadcast(const Ta *a, const Tb *b,
    unsigned int k, unsigned int m, unsigned int n, Tc *c) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, k*m*n) {
    if (broadcast_inner) {
      c[idx] = op(a[idx], b[(idx / m / n) * n  + (idx % n)]);
    } else {
      c[idx] = op(a[idx], b[(idx / n) % m]);
    }
  }
}

template<typename Ta, typename Tb, typename Op, bool broadcast_inner>
__global__ void kernel_binary_broadcast_inplace(Ta *a, const Tb *b,
    unsigned int k, unsigned int m, unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, k*m*n) {
    if (broadcast_inner) {
      a[idx] = op(a[idx], b[(idx / m / n) * n  + (idx % n)]);
    } else {
      a[idx] = op(a[idx], b[(idx / n) % m]);
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
      binary_broadcast<Ta, Tb, Tc, AddOp<Ta, Tb, Tc> >
          (btype, a, b, k, m, n, c);
      break;
    case DIV_OP:
      binary_broadcast<Ta, Tb, Tc, DivOp<Ta, Tb, Tc> >
          (btype, a, b, k, m, n, c);
      break;
    case MAX_B_OP:
      binary_broadcast<Ta, Tb, Tc, MaxOp<Ta, Tb, Tc> >(
          btype, a, b, k, m, n, c);
      break;
    case MIN_B_OP:
      binary_broadcast<Ta, Tb, Tc, MinOp<Ta, Tb, Tc> >
          (btype, a, b, k, m, n, c);
      break;
    case MUL_OP:
      binary_broadcast<Ta, Tb, Tc, MulOp<Ta, Tb, Tc> >
          (btype, a, b, k, m, n, c);
      break;
    case POW_OP:
      binary_broadcast<Ta, Tb, Tc, PowOp<Ta, Tb, Tc> >
          (btype, a, b, k, m, n, c);
      break;
    case SUB_OP:
      binary_broadcast<Ta, Tb, Tc, SubOp<Ta, Tb, Tc> >
          (btype, a, b, k, m, n, c);
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



BINARY_OP(EqOp, a == b)
BINARY_OP(GtOp, a > b)
BINARY_OP(GtEqOp, a >= b)
BINARY_OP(LtOp, a < b)
BINARY_OP(LtEqOp, a <= b)
BINARY_OP(NeqOp, a != b)


template<typename Ta, typename Tb>
void binary_cmp(BinaryCmpOp op, const Ta *a, const Tb *b, unsigned int n,
                bool_t *c) {
  switch (op) {
    case EQ_OP:
      binary<Ta, Tb, bool_t, EqOp<Ta, Tb, bool_t> >(a, b, n, c);
      break;
    case GT_OP:
      binary<Ta, Tb, bool_t, GtOp<Ta, Tb, bool_t> >(a, b, n, c);
      break;
    case GT_EQ_OP:
      binary<Ta, Tb, bool_t, GtEqOp<Ta, Tb, bool_t> >(a, b, n, c);
      break;
    case LT_OP:
      binary<Ta, Tb, bool_t, LtOp<Ta, Tb, bool_t> >(a, b, n, c);
      break;
    case LT_EQ_OP:
      binary<Ta, Tb, bool_t, LtEqOp<Ta, Tb, bool_t> >(a, b, n, c);
      break;
    case NEQ_OP:
      binary<Ta, Tb, bool_t, NeqOp<Ta, Tb, bool_t> >(a, b, n, c);
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



template<typename T>
void binary_cmp_scalar(BinaryCmpOp op, const T *a, T alpha,
                       unsigned int n, bool_t *b) {
  switch (op) {
    case EQ_OP:
      binary_scalar<T, T, bool_t, EqOp<T, T, bool_t> >
                   (a, alpha, n, b);
      break;
    case GT_OP:
      binary_scalar<T, T, bool_t, GtOp<T, T, bool_t> >
                   (a, alpha, n, b);
      break;
    case GT_EQ_OP:
      binary_scalar<T, T, bool_t, GtEqOp<T, T, bool_t> >
                   (a, alpha, n, b);
      break;
    case LT_OP:
      binary_scalar<T, T, bool_t, LtOp<T, T, bool_t> >
                   (a, alpha, n, b);
      break;
    case LT_EQ_OP:
      binary_scalar<T, T, bool_t, LtEqOp<T, T, bool_t> >
                   (a, alpha, n, b);
      break;
    case NEQ_OP:
      binary_scalar<T, T, bool_t, NeqOp<T, T, bool_t> >
                   (a, alpha, n, b);
      break;
  }
}

template void binary_cmp_scalar<float>(
    BinaryCmpOp op, const float *a, float alpha, unsigned int n, bool_t *b);
template void binary_cmp_scalar<int>(
    BinaryCmpOp op, const int *a, int alpha, unsigned int n, bool_t *b);




template<typename T>
void binary_cmp_scalar_(BinaryCmpOp op, T alpha, const T *a,
                       unsigned int n, bool_t *b) {
  switch (op) {
    case EQ_OP:
      binary_scalar<T, T, bool_t, EqOp<T, T, bool_t> >
          (alpha, a, n, b);
      break;
    case GT_OP:
      binary_scalar<T, T, bool_t, GtOp<T, T, bool_t> >
          (alpha, a, n, b);
      break;
    case GT_EQ_OP:
      binary_scalar<T, T, bool_t, GtEqOp<T, T, bool_t> >
          (alpha, a, n, b);
      break;
    case LT_OP:
      binary_scalar<T, T, bool_t, LtOp<T, T, bool_t> >
          (alpha, a, n, b);
      break;
    case LT_EQ_OP:
      binary_scalar<T, T, bool_t, LtEqOp<T, T, bool_t> >
          (alpha, a, n, b);
      break;
    case NEQ_OP:
      binary_scalar<T, T, bool_t, NeqOp<T, T, bool_t> >
          (alpha, a, n, b);
      break;
  }
}

template void binary_cmp_scalar_<float>(
    BinaryCmpOp op, float alpha, const float *a, unsigned int n, bool_t *b);;
template void binary_cmp_scalar_<int>(
    BinaryCmpOp op, int alpha, const int *a, unsigned int n, bool_t *b);




template<typename Ta, typename Tb>
void binary_cmp_broadcast(BinaryCmpOp op, BroadcastType btype, const Ta *a,
    const Tb *b, unsigned int k, unsigned int m, unsigned int n, bool_t *c) {
  switch (op) {
    case EQ_OP:
      binary_broadcast<Ta, Tb, bool_t, EqOp<Ta, Tb, bool_t> >
                      (btype, a, b, k, m, n, c);
      break;
    case GT_OP:
      binary_broadcast<Ta, Tb, bool_t, GtOp<Ta, Tb, bool_t> >
                      (btype, a, b, k, m, n, c);
      break;
    case GT_EQ_OP:
      binary_broadcast<Ta, Tb, bool_t, GtEqOp<Ta, Tb, bool_t> >
                      (btype, a, b, k, m, n, c);
      break;
    case LT_OP:
      binary_broadcast<Ta, Tb, bool_t, LtOp<Ta, Tb, bool_t> >
                      (btype, a, b, k, m, n, c);
      break;
    case LT_EQ_OP:
      binary_broadcast<Ta, Tb, bool_t, LtEqOp<Ta, Tb, bool_t> >
                      (btype, a, b, k, m, n, c);
      break;
    case NEQ_OP:
      binary_broadcast<Ta, Tb, bool_t, NeqOp<Ta, Tb, bool_t> >
                      (btype, a, b, k, m, n, c);
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





#define UNARY_OP(name, operation) \
template <typename Ta, typename Tb> \
struct name { \
  __device__ Tb operator()(const Ta a) { \
    operation; \
  } \
};

UNARY_OP(AbsOp, return fabsf(a);)
UNARY_OP(CosOp, return cosf(a);)
UNARY_OP(ExpOp, return expf(a);)
UNARY_OP(LogOp, return logf(a);)
UNARY_OP(Log1pOp, return log1pf(a);)
UNARY_OP(NegOp, return -a;)
UNARY_OP(ReluOp, return fmaxf(0.0, a);)
UNARY_OP(ReluDOp, return a >= 0.0 ? 1.0 : 0.0;)
UNARY_OP(SigmoidOp, return 1.0/(1.0 + expf(-a));)
UNARY_OP(SigmoidDOp, Ta tmp = 1.0/(1.0 + expf(-a)); return tmp*(1-tmp);)
UNARY_OP(SinOp, return sinf(a);)
UNARY_OP(SoftplusOp, return a > 25.0 ? a : log1pf(expf(a));)
UNARY_OP(SoftplusDOp, return a > 25.0 ? 1.0 : 1.0 - 1.0/(1.0 + expf(a));)
UNARY_OP(SqrtOp, return sqrtf(a);)
UNARY_OP(TanhOp, return tanhf(a);)
UNARY_OP(TanhDOp, Ta tmp = tanhf(a); return 1-tmp*tmp;)



template<typename T, typename Op>
__global__ void kernel_unary(const T *a, unsigned int n, T *b) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = op(a[idx]);
  }
}

template<typename T, typename Op>
__global__ void kernel_unary_inplace(T *a, unsigned int n) {
  Op op;
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = op(a[idx]);
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
      unary<T, AbsOp<T, T> >(a, n, b);
      break;
    case COS_OP:
      unary<T, CosOp<T, T> >(a, n, b);
      break;
    case EXP_OP:
      unary<T, ExpOp<T, T> >(a, n, b);
      break;
    case LOG_OP:
      unary<T, LogOp<T, T> >(a, n, b);
      break;
    case LOG1P_OP:
      unary<T, Log1pOp<T, T> >(a, n, b);
      break;
    case NEG_OP:
      unary<T, NegOp<T, T> >(a, n, b);
      break;
    case RELU_OP:
      unary<T, ReluOp<T, T> >(a, n, b);
      break;
    case RELU_D_OP:
      unary<T, ReluDOp<T, T> >(a, n, b);
      break;
    case SIGMOID_OP:
      unary<T, SigmoidOp<T, T> >(a, n, b);
      break;
    case SIGMOID_D_OP:
      unary<T, SigmoidDOp<T, T> >(a, n, b);
      break;
    case SIN_OP:
      unary<T, SinOp<T, T> >(a, n, b);
      break;
    case SOFTPLUS_OP:
      unary<T, SoftplusOp<T, T> >(a, n, b);
      break;
    case SOFTPLUS_D_OP:
      unary<T, SoftplusDOp<T, T> >(a, n, b);
      break;
    case SQRT_OP:
      unary<T, SqrtOp<T, T> >(a, n, b);
      break;
    case TANH_OP:
      unary<T, TanhOp<T, T> >(a, n, b);
      break;
    case TANH_D_OP:
      unary<T, TanhDOp<T, T> >(a, n, b);
      break;
  }
}
template void unary<float>(UnaryOp op, const float *a, unsigned int n,
                           float *b);
// TODO: unary should convert to float for certain operations
template void unary<int>(UnaryOp op, const int *a, unsigned int n, int *b);





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
