#ifndef BLAS_HPP_
#define BLAS_HPP_

#include <cublas_v2.h>


namespace cudarray {
// TODO: implement more from
// http://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf

enum TransposeOp {
  OP_TRANS = CUBLAS_OP_T,
  OP_NO_TRANS = CUBLAS_OP_N
};

template<typename T>
T dot(const T *a, const T *b, unsigned int n);

template<typename T>
void gemv(const T *A, const T *b, TransposeOp trans, unsigned int m,
          unsigned int n, T alpha, T beta, T *c);

template<typename T>
void gemm(const T *A, const T *B, TransposeOp transA, TransposeOp transB,
          unsigned int m, unsigned int n, unsigned int k, T alpha, T beta,
          T *C);


template <typename T>
class BLASBatch {
public:
  BLASBatch(const T **A, const T **B, T **C, unsigned int batch_size);
  BLASBatch(const T *A, const T *B, T *C, unsigned int batch_size, int Astride,
            int Bstride, int Cstride);

  ~BLASBatch();

  void gemm(TransposeOp transA, TransposeOp transB, unsigned int m,
            unsigned int n, unsigned int k, T alpha, T beta);
private:
  unsigned int batch_size;
  const float **As_dev;
  const float **Bs_dev;
  float **Cs_dev;
};

}

#endif // BLAS_HPP_
