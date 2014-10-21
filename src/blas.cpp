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


template <typename T>
T **dev_ptrs(const T *base, int num, int stride) {
  T *ptrs_host[num];
  int idx = 0;
  for(int n = 0; n < num; ++n){
    ptrs_host[idx] = (T *) base + n * stride;
    idx++;
  }
  T **ptrs_dev;
  CUDA_CHECK(cudaMalloc((void **) &ptrs_dev, num*sizeof(T *)));
  CUDA_CHECK(cudaMemcpy(ptrs_dev, ptrs_host, num*sizeof(T *),
                        cudaMemcpyHostToDevice));
  return ptrs_dev;
}


template <typename T>
BLASBatch<T>::BLASBatch(const T **As, const T **Bs, T **Cs,
    unsigned int batch_size) : batch_size(batch_size) {
  size_t ptrs_size = batch_size * sizeof(T **);
  CUDA_CHECK(cudaMalloc((void **) &As_dev, ptrs_size));
  CUDA_CHECK(cudaMemcpy(As_dev, As, batch_size*sizeof(float *),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void **) &Bs_dev, ptrs_size));
  CUDA_CHECK(cudaMemcpy(Bs_dev, Bs, batch_size*sizeof(float *),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void **) &Cs_dev, ptrs_size));
  CUDA_CHECK(cudaMemcpy(Cs_dev, Cs, batch_size*sizeof(float *),
                        cudaMemcpyHostToDevice));
}


template <typename T>
BLASBatch<T>::BLASBatch(const T *A, const T *B, T *C,
    unsigned int batch_size, int Astride, int Bstride, int Cstride)
    : batch_size(batch_size) {
  As_dev = (const float **) dev_ptrs(A, batch_size, Astride);
  Bs_dev = (const float **) dev_ptrs(B, batch_size, Bstride);
  Cs_dev = dev_ptrs(C, batch_size, Cstride);
}

template <typename T>
BLASBatch<T>::~BLASBatch() {
  CUDA_CHECK(cudaFree(As_dev));
  CUDA_CHECK(cudaFree(Bs_dev));
  CUDA_CHECK(cudaFree(Cs_dev));
}


cublasStatus_t cublas_gemm_batched(cublasHandle_t handle,
    cublasOperation_t transA, cublasOperation_t transB, int m, int n, int k,
    const float *alpha, const float *Aarray[], int lda, const float *Barray[],
    int ldb, const float *beta, float *Carray[], int ldc, int batchCount) {
  return cublasSgemmBatched(handle, transA, transB, m, n, k, alpha, Aarray,
      lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

template <typename T>
void BLASBatch<T>::gemm(TransposeOp transA, TransposeOp transB, unsigned int m,
                        unsigned int n, unsigned int k, T alpha, T beta) {
  int lda = (transA == OP_NO_TRANS) ? k : m;
  int ldb = (transB == OP_NO_TRANS) ? n : k;
  int ldc = n;
  cublasOperation_t cuTransA = (cublasOperation_t) transA;
  cublasOperation_t cuTransB = (cublasOperation_t) transB;
  CUBLAS_CHECK(cublas_gemm_batched(CUDA::cublas_handle(), cuTransB, cuTransA,
      n, m, k, &alpha, Bs_dev, ldb, As_dev, lda, &beta, Cs_dev, ldc,
      batch_size));
}

template class BLASBatch<float>;

}
