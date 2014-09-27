#include <cublas_v2.h>
#include "cudarray/common.hpp"
#include "cudarray/blas.hpp"

namespace cudarray {


cublasStatus_t cublas_dot(cublasHandle_t handle, int n, const float *x,
    int incx, const float *y, int incy, float *result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}

template<typename T>
T dot(const T *a, const T *b, unsigned int n) {
  T result;
  CUBLAS_CHECK(cublas_dot(CUDA::cublas_handle(), n, a, 1, b, 1, &result));
  return result;
}

template float dot<float>(const float *x, const float *y, unsigned int n);



cublasStatus_t cublas_gemv(cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, const float *alpha, const float *A, int lda, const float *x,
    int incx, const float *beta, float *y, int incy) {
  return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                     incy);
}

template<typename T>
void gemv(const T *A, const T *b, TransposeOp trans, unsigned int m,
          unsigned int n, T alpha, T beta, T *c) {
  cublasOperation_t cuTrans;
  if (trans == OP_TRANS) {
    cuTrans = CUBLAS_OP_N;
    unsigned int tmp = n;
    n = m;
    m = tmp;
  } else {
    cuTrans = CUBLAS_OP_T;
  }
  int lda = n;
  CUBLAS_CHECK(cublas_gemv(CUDA::cublas_handle(), cuTrans, n, m, &alpha, A,
     lda, b, 1, &beta, c, 1));
}

template void gemv<float>(const float *A, const float *b, TransposeOp trans,
    unsigned int m, unsigned int n, float alpha, float beta, float *c);



cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transA,
    cublasOperation_t transB, int m, int n, int k, const float *alpha,
    const float *A, int lda, const float *B, int ldb, const float *beta,
    float *C, int ldc) {
  return cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

template <typename T>
void gemm(const T *A, const T *B, TransposeOp transA, TransposeOp transB,
          unsigned int m, unsigned int n, unsigned int k, T alpha, T beta,
          T *C) {
  int lda = (transA == OP_NO_TRANS) ? k : m;
  int ldb = (transB == OP_NO_TRANS) ? n : k;
  int ldc = n;
  cublasOperation_t cuTransA = (cublasOperation_t) transA;
  cublasOperation_t cuTransB = (cublasOperation_t) transB;
  CUBLAS_CHECK(cublas_gemm(CUDA::cublas_handle(), cuTransB, cuTransA,
                           n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

template void gemm<float>(const float *A, const float *B, TransposeOp transA,
    TransposeOp transB, unsigned int m, unsigned int n, unsigned int k,
    float alpha, float beta, float *C);

}
