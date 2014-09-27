#include "cudarray/common.hpp"
#include "cudarray/elementwise.hpp"


namespace cudarray {

template<typename T, BinaryOp op>
__global__ void kernel_binary(const T *a, const T *b, unsigned int n, T *c) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    if (op == ADD_OP) c[idx] = a[idx] + b[idx];
    if (op == DIV_OP) c[idx] = a[idx] / b[idx];
    if (op == MAX_B_OP) c[idx] = fmaxf(a[idx], b[idx]);
    if (op == MIN_B_OP) c[idx] = fminf(a[idx], b[idx]);
    if (op == MUL_OP) c[idx] = a[idx] * b[idx];
    if (op == POW_OP) c[idx] = powf(a[idx], b[idx]);
    if (op == SUB_OP) c[idx] = a[idx] - b[idx];
  }
}

template<typename T, BinaryOp op>
void binary(const T *a, const T *b, unsigned int n, T *c) {
  kernel_binary<T, op><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, b, n, c);
}

template<typename T>
void binary(BinaryOp op, const T *a, const T *b, unsigned int n, T *c) {
  switch (op) {
    case ADD_OP:
      binary<T, ADD_OP>(a, b, n, c);
      break;
    case DIV_OP:
      binary<T, DIV_OP>(a, b, n, c);
      break;
    case MAX_B_OP:
      binary<T, MAX_B_OP>(a, b, n, c);
      break;
    case MIN_B_OP:
      binary<T, MIN_B_OP>(a, b, n, c);
      break;
    case MUL_OP:
      binary<T, MUL_OP>(a, b, n, c);
      break;
    case POW_OP:
      binary<T, POW_OP>(a, b, n, c);
      break;
    case SUB_OP:
      binary<T, SUB_OP>(a, b, n, c);
      break;
  }
}
template void binary<float>(BinaryOp op, const float *a, const float *b,
                            unsigned int n, float *c);




template<typename T, BinaryOp op>
__global__ void kernel_binary_inplace(T *a, const T *b, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    if (op == ADD_OP) a[idx] += b[idx];
    if (op == DIV_OP) a[idx] /= b[idx];
    if (op == MAX_B_OP) a[idx] = fmaxf(a[idx], b[idx]);
    if (op == MIN_B_OP) a[idx] = fminf(a[idx], b[idx]);
    if (op == MUL_OP) a[idx] *= b[idx];
    if (op == POW_OP) a[idx] = powf(a[idx], b[idx]);
    if (op == SUB_OP) a[idx] -= b[idx];
  }
}

template<typename T, BinaryOp op>
void binary_inplace(T *a, const T *b, unsigned int n) {
  kernel_binary_inplace<T, op><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
      (a, b, n);
}

template<typename T>
void binary_inplace(BinaryOp op, T *a, const T *b, unsigned int n) {
  switch (op) {
    case ADD_OP:
      binary_inplace<T, ADD_OP>(a, b, n);
      break;
    case DIV_OP:
      binary_inplace<T, DIV_OP>(a, b, n);
      break;
    case MAX_B_OP:
      binary_inplace<T, MAX_B_OP>(a, b, n);
      break;
    case MIN_B_OP:
      binary_inplace<T, MIN_B_OP>(a, b, n);
      break;
    case MUL_OP:
      binary_inplace<T, MUL_OP>(a, b, n);
      break;
    case POW_OP:
      binary_inplace<T, POW_OP>(a, b, n);
      break;
    case SUB_OP:
      binary_inplace<T, SUB_OP>(a, b, n);
      break;
  }
}
template void binary_inplace<float>(BinaryOp op, float *a, const float *b,
                                    unsigned int n);




template<typename T, BinaryOp op>
__global__ void kernel_binary_scalar(const T *a, T alpha, unsigned int n,
                                     T *c) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    if (op == ADD_OP) c[idx] = a[idx] + alpha;
    if (op == DIV_OP) c[idx] = a[idx] / alpha;
    if (op == MAX_B_OP) c[idx] = fmaxf(a[idx], alpha);
    if (op == MIN_B_OP) c[idx] = fminf(a[idx], alpha);
    if (op == MUL_OP) c[idx] = a[idx] * alpha;
    if (op == POW_OP) c[idx] = powf(a[idx], alpha);
    if (op == SUB_OP) c[idx] = a[idx] - alpha;
  }
}

template<typename T, BinaryOp op>
void binary_scalar(const T *a, T alpha, unsigned int n, T *c) {
  kernel_binary_scalar<T, op><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
      (a, alpha, n, c);
}

template<typename T>
void binary_scalar(BinaryOp op, const T *a, T alpha, unsigned int n, T *c) {
  switch (op) {
    case ADD_OP:
      binary_scalar<T, ADD_OP>(a, alpha, n, c);
      break;
    case DIV_OP:
      binary_scalar<T, DIV_OP>(a, alpha, n, c);
      break;
    case MAX_B_OP:
      binary_scalar<T, MAX_B_OP>(a, alpha, n, c);
      break;
    case MIN_B_OP:
      binary_scalar<T, MIN_B_OP>(a, alpha, n, c);
      break;
    case MUL_OP:
      binary_scalar<T, MUL_OP>(a, alpha, n, c);
      break;
    case POW_OP:
      binary_scalar<T, POW_OP>(a, alpha, n, c);
      break;
    case SUB_OP:
      binary_scalar<T, SUB_OP>(a, alpha, n, c);
      break;
  }
}
template void binary_scalar<float>(BinaryOp op, const float *a, float alpha,
                            unsigned int n, float *c);





template<typename T, BinaryOp op>
__global__ void kernel_binary_scalar_inplace(T *a, T alpha, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    if (op == ADD_OP) a[idx] += alpha;
    if (op == DIV_OP) a[idx] /= alpha;
    if (op == MAX_B_OP) a[idx] = fmaxf(a[idx], alpha);
    if (op == MIN_B_OP) a[idx] = fminf(a[idx], alpha);
    if (op == MUL_OP) a[idx] *= alpha;
    if (op == POW_OP) a[idx] = powf(a[idx], alpha);
    if (op == SUB_OP) a[idx] -= alpha;
  }
}

template<typename T, BinaryOp op>
void binary_scalar_inplace(T *a, T alpha, unsigned int n) {
  kernel_binary_scalar_inplace<T, op><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
      (a, alpha, n);
}

template<typename T>
void binary_scalar_inplace(BinaryOp op, T *a, T alpha, unsigned int n) {
  switch (op) {
    case ADD_OP:
      binary_scalar_inplace<T, ADD_OP>(a, alpha, n);
      break;
    case DIV_OP:
      binary_scalar_inplace<T, DIV_OP>(a, alpha, n);
      break;
    case MAX_B_OP:
      binary_scalar_inplace<T, MAX_B_OP>(a, alpha, n);
      break;
    case MIN_B_OP:
      binary_scalar_inplace<T, MIN_B_OP>(a, alpha, n);
      break;
    case MUL_OP:
      binary_scalar_inplace<T, MUL_OP>(a, alpha, n);
      break;
    case POW_OP:
      binary_scalar_inplace<T, POW_OP>(a, alpha, n);
      break;
    case SUB_OP:
      binary_scalar_inplace<T, SUB_OP>(a, alpha, n);
      break;
  }
}
template void binary_scalar_inplace<float>(BinaryOp op, float *a, float alpha,
                                           unsigned int n);





template<typename T, BinaryOp op, bool broadcast_to_leading>
__global__ void kernel_binary_broadcast(const T *a, const T *b,
    unsigned int m, unsigned int n, T *c) {
  CUDA_GRID_STRIDE_LOOP(idx, m*n) {
    if (broadcast_to_leading) {
      if (op == ADD_OP) c[idx] = a[idx] + b[idx % n];
      if (op == DIV_OP) c[idx] = a[idx] / b[idx % n];
      if (op == MAX_B_OP) c[idx] = fmaxf(a[idx], b[idx % n]);
      if (op == MIN_B_OP) c[idx] = fminf(a[idx], b[idx % n]);
      if (op == MUL_OP) c[idx] = a[idx] * b[idx % n];
      if (op == POW_OP) c[idx] = powf(a[idx], b[idx % n]);
      if (op == SUB_OP) c[idx] = a[idx] - b[idx % n];
    } else {
      if (op == ADD_OP) c[idx] = a[idx] + b[idx / m];
      if (op == DIV_OP) c[idx] = a[idx] / b[idx / m];
      if (op == MAX_B_OP) c[idx] = fmaxf(a[idx], b[idx / m]);
      if (op == MIN_B_OP) c[idx] = fminf(a[idx], b[idx / m]);
      if (op == MUL_OP) c[idx] = a[idx] * b[idx / m];
      if (op == POW_OP) c[idx] = powf(a[idx], b[idx / m]);
      if (op == SUB_OP) c[idx] = a[idx] - b[idx / m];
    }
  }
}

template<typename T, BinaryOp op>
void binary_broadcast(const T *a, const T *b, unsigned int m,
                      unsigned int n, bool broadcast_to_leading, T *c) {
  if (broadcast_to_leading) {
    kernel_binary_broadcast<T, op, true>
        <<<CUDA_BLOCKS(m*n), CUDA_NUM_THREADS>>>
        (a, b, m, n, c);
  } else {
    kernel_binary_broadcast<T, op, false>
        <<<CUDA_BLOCKS(m*n), CUDA_NUM_THREADS>>>
        (a, b, m, n, c);
  }
}

template<typename T>
void binary_broadcast(BinaryOp op, const T *a, const T *b, unsigned int m,
                      unsigned int n, bool broadcast_to_leading, T *c) {
  switch (op) {
    case ADD_OP:
      binary_broadcast<T, ADD_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
    case DIV_OP:
      binary_broadcast<T, DIV_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
    case MAX_B_OP:
      binary_broadcast<T, MAX_B_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
    case MIN_B_OP:
      binary_broadcast<T, MIN_B_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
    case MUL_OP:
      binary_broadcast<T, MUL_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
    case POW_OP:
      binary_broadcast<T, POW_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
    case SUB_OP:
      binary_broadcast<T, SUB_OP>(a, b, m, n, broadcast_to_leading, c);
      break;
  }
}
template void binary_broadcast<float>(BinaryOp op, const float *a,
    const float *b, unsigned int m, unsigned int n, bool broadcast_to_leading,
    float *c);







template<typename T, BinaryOp op, bool broadcast_to_leading>
__global__ void kernel_binary_broadcast_inplace(T *a, const T *b,
    unsigned int m, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, m*n) {
    if (broadcast_to_leading) {
      if (op == ADD_OP) a[idx] += b[idx % n];
      if (op == DIV_OP) a[idx] /= b[idx % n];
      if (op == MAX_B_OP) a[idx] = fmaxf(a[idx], b[idx % n]);
      if (op == MIN_B_OP) a[idx] = fminf(a[idx], b[idx % n]);
      if (op == MUL_OP) a[idx] *= b[idx % n];
      if (op == POW_OP) a[idx] = powf(a[idx], b[idx % n]);
      if (op == SUB_OP) a[idx] -= b[idx % n];
    } else {
      if (op == ADD_OP) a[idx] += b[idx / m];
      if (op == DIV_OP) a[idx] /= b[idx / m];
      if (op == MAX_B_OP) a[idx] = fmaxf(a[idx], b[idx / m]);
      if (op == MIN_B_OP) a[idx] = fminf(a[idx], b[idx / m]);
      if (op == MUL_OP) a[idx] *= b[idx / m];
      if (op == POW_OP) a[idx] = powf(a[idx], b[idx / m]);
      if (op == SUB_OP) a[idx] -= b[idx / m];
    }
  }
}

template<typename T, BinaryOp op>
void binary_broadcast_inplace(T *a, const T *b, unsigned int m,
                      unsigned int n, bool broadcast_to_leading) {
  if (broadcast_to_leading) {
    kernel_binary_broadcast_inplace<T, op, true>
        <<<CUDA_BLOCKS(m*n), CUDA_NUM_THREADS>>>
        (a, b, m, n);
  } else {
    kernel_binary_broadcast_inplace<T, op, false>
        <<<CUDA_BLOCKS(m*n), CUDA_NUM_THREADS>>>
        (a, b, m, n);
  }
}

template<typename T>
void binary_broadcast_inplace(BinaryOp op, T *a, const T *b, unsigned int m,
                      unsigned int n, bool broadcast_to_leading) {
  switch (op) {
    case ADD_OP:
      binary_broadcast_inplace<T, ADD_OP>(a, b, m, n, broadcast_to_leading);
      break;
    case DIV_OP:
      binary_broadcast_inplace<T, DIV_OP>(a, b, m, n, broadcast_to_leading);
      break;
    case MAX_B_OP:
      binary_broadcast_inplace<T, MAX_B_OP>(a, b, m, n, broadcast_to_leading);
      break;
    case MIN_B_OP:
      binary_broadcast_inplace<T, MIN_B_OP>(a, b, m, n, broadcast_to_leading);
      break;
    case MUL_OP:
      binary_broadcast_inplace<T, MUL_OP>(a, b, m, n, broadcast_to_leading);
      break;
    case POW_OP:
      binary_broadcast_inplace<T, POW_OP>(a, b, m, n, broadcast_to_leading);
      break;
    case SUB_OP:
      binary_broadcast_inplace<T, SUB_OP>(a, b, m, n, broadcast_to_leading);
      break;
  }
}
template void binary_broadcast_inplace<float>(BinaryOp op, float *a,
    const float *b, unsigned int m, unsigned int n, bool broadcast_to_leading);






template<typename T, UnaryOp op>
__global__ void kernel_unary(const T *a, unsigned int n, T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    if (op == ABS_OP) b[idx] = fabsf(a[idx]);
    if (op == EXP_OP) b[idx] = expf(a[idx]);
    if (op == LOG_OP) b[idx] = logf(a[idx]);
    if (op == NEG_OP) b[idx] = -a[idx];
    if (op == RELU_OP) b[idx] = fmaxf(0.0, a[idx]);
    if (op == RELU_D_OP) b[idx] = 1.0/(1.0 + expf(-a[idx]));
    if (op == SIGMOID_OP) b[idx] = 1.0/(1.0 + expf(-a[idx]));
    if (op == SIGMOID_D_OP) {
      T tmp = 1.0/(1.0 + expf(-a[idx]));
      b[idx] = tmp*(1-tmp);
    }
    if (op == SQRT_OP) b[idx] = sqrtf(a[idx]);
    if (op == TANH_OP) b[idx] = tanhf(a[idx]);
    if (op == TANH_D_OP) {
      T tmp = expf(2.0*a[idx]);
      b[idx] = (tmp-1.0)/(tmp+1.0);
    }
  }
}

template<typename T, UnaryOp op>
void unary(const T *a, unsigned int n, T *b) {
  kernel_unary<T, op><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, n, b);
}

template<typename T>
void unary(UnaryOp op, const T *a, unsigned int n, T *b) {
  switch (op) {
    case ABS_OP:
      unary<T, ABS_OP>(a, n, b);
      break;
    case EXP_OP:
      unary<T, EXP_OP>(a, n, b);
      break;
    case LOG_OP:
      unary<T, LOG_OP>(a, n, b);
      break;
    case NEG_OP:
      unary<T, NEG_OP>(a, n, b);
      break;
    case RELU_OP:
      unary<T, RELU_OP>(a, n, b);
      break;
    case RELU_D_OP:
      unary<T, RELU_D_OP>(a, n, b);
      break;
    case SIGMOID_OP:
      unary<T, SIGMOID_OP>(a, n, b);
      break;
    case SIGMOID_D_OP:
      unary<T, SIGMOID_D_OP>(a, n, b);
      break;
    case SQRT_OP:
      unary<T, SQRT_OP>(a, n, b);
      break;
    case TANH_OP:
      unary<T, TANH_OP>(a, n, b);
      break;
    case TANH_D_OP:
      unary<T, TANH_D_OP>(a, n, b);
      break;
  }
}
template void unary<float>(UnaryOp op, const float *a, unsigned int n,
                           float *b);



template<typename T, UnaryOp op>
__global__ void kernel_unary_inplace(T *a, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    if (op == ABS_OP) a[idx] = fabsf(a[idx]);
    if (op == EXP_OP) a[idx] = expf(a[idx]);
    if (op == LOG_OP) a[idx] = logf(a[idx]);
    if (op == NEG_OP) a[idx] = -a[idx];
    if (op == RELU_OP) a[idx] = fmaxf(0.0, a[idx]);
    if (op == RELU_D_OP) a[idx] = 1.0/(1.0 + expf(-a[idx]));
    if (op == SIGMOID_OP) a[idx] = 1.0/(1.0 + expf(-a[idx]));
    if (op == SIGMOID_D_OP) {
      T tmp = 1.0/(1.0 + expf(-a[idx]));
      a[idx] = tmp*(1-tmp);
    }
    if (op == SQRT_OP) a[idx] = sqrtf(a[idx]);
    if (op == TANH_OP) a[idx] = tanhf(a[idx]);
    if (op == TANH_D_OP) {
      T tmp = expf(2.0*a[idx]);
      a[idx] = (tmp-1.0)/(tmp+1.0);
    }
  }
}

template<typename T, UnaryOp op>
void unary_inplace(T *a, unsigned int n) {
  kernel_unary_inplace<T, op><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, n);
}

template<typename T>
void unary_inplace(UnaryOp op, T *a, unsigned int n) {
  switch (op) {
    case ABS_OP:
      unary_inplace<T, ABS_OP>(a, n);
      break;
    case EXP_OP:
      unary_inplace<T, EXP_OP>(a, n);
      break;
    case LOG_OP:
      unary_inplace<T, LOG_OP>(a, n);
      break;
    case NEG_OP:
      unary_inplace<T, NEG_OP>(a, n);
      break;
    case RELU_OP:
      unary_inplace<T, RELU_OP>(a, n);
      break;
    case RELU_D_OP:
      unary_inplace<T, RELU_D_OP>(a, n);
      break;
    case SIGMOID_OP:
      unary_inplace<T, SIGMOID_OP>(a, n);
      break;
    case SIGMOID_D_OP:
      unary_inplace<T, SIGMOID_D_OP>(a, n);
      break;
    case SQRT_OP:
      unary_inplace<T, SQRT_OP>(a, n);
      break;
    case TANH_OP:
      unary_inplace<T, TANH_OP>(a, n);
      break;
    case TANH_D_OP:
      unary_inplace<T, TANH_D_OP>(a, n);
      break;
  }
}

template void unary_inplace<float>(UnaryOp op, float *a, unsigned int n);




template<typename T>
__global__ void kernel_clip(const T *a, T a_min, T a_max, unsigned int n,
                            T *b) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    b[idx] = fminf(fmaxf(a[idx], a_min), a_max);
  }
}

template<typename T>
void clip(const T *a, T a_min, T a_max, unsigned int n, T *b) {
  kernel_clip<T><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>(a, a_min, a_max, n, b);
}

template void clip<float>(const float *a, float a_min, float a_max,
                          unsigned int n, float *b);

template<typename T>
__global__ void kernel_clip_inplace(T *a, T a_min, T a_max, unsigned int n) {
  CUDA_GRID_STRIDE_LOOP(idx, n) {
    a[idx] = fminf(fmaxf(a[idx], a_min), a_max);
  }
}

template<typename T>
void clip_inplace(T *a, T a_min, T a_max, unsigned int n) {
  kernel_clip_inplace<T><<<CUDA_BLOCKS(n), CUDA_NUM_THREADS>>>
      (a, a_min, a_max, n);
}

template void clip_inplace<float>(float *a, float a_min, float a_max,
                                  unsigned int n);

}
