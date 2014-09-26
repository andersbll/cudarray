#ifndef ELEMENTWISE_HPP_
#define ELEMENTWISE_HPP_


namespace cudarray {
// TODO: implement more from
// http://docs.nvidia.com/cuda/pdf/CUDA_Math_API.pdf

enum BinaryOp {
  ADD_OP, DIV_OP, MAX_B_OP, MIN_B_OP, MUL_OP, POW_OP, SUB_OP
};

template<typename T>
void binary(BinaryOp op, const T *a, const T *b, unsigned int n, T *c);
template<typename T>
void binary_inplace(BinaryOp op, T *a, const T *b, unsigned int n);
template<typename T>
void binary_scalar(BinaryOp op, const T *a, T alpha, unsigned int n, T *b);
template<typename T>
void binary_scalar_inplace(BinaryOp op, T *a, T alpha, unsigned int n);
template<typename T>
void binary_broadcast(BinaryOp op, const T *a, const T *b, unsigned int m,
                      unsigned int n, bool broadcast_to_leading, T *c);
template<typename T>
void binary_broadcast_inplace(BinaryOp op, T *a, const T *b, unsigned int m,
                              unsigned int n, bool broadcast_to_leading);


enum UnaryOp {
  ABS_OP, EXP_OP, LOG_OP, RELU_OP, RELU_D_OP, SIGMOID_OP, SIGMOID_D_OP,
  SQRT_OP, TANH_OP, TANH_D_OP
};

template<typename T>
void unary(UnaryOp op, const T *a, unsigned int n, T *b);
template<typename T>
void unary_inplace(UnaryOp op, T *a, unsigned int n);

}

#endif // ELEMENTWISE_HPP_
