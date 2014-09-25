#ifndef ELEMENTWISE_HPP_
#define ELEMENTWISE_HPP_


namespace cudarray {
// TODO: implement more from
// http://docs.nvidia.com/cuda/pdf/CUDA_Math_API.pdf

#define BINARY_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, const T *b, unsigned int n, T *c); \
  template<typename T> \
  void name##_inplace(T *a, const T *b, unsigned int n); \
  template<typename T> \
  void name##_scalar(const T *a, T alpha, unsigned int n, T *b); \
  template<typename T> \
  void name##_scalar_inplace(T *a, T alpha, unsigned int n); \
  template<typename T> \
  void name##_broadcast(const T *a, const T *b, unsigned int m, \
                        unsigned int n, bool broadcast_to_leading, T *c); \
  template<typename T> \
  void name##_broadcast_inplace(T *a, const T *b, unsigned int m, \
                                unsigned int n, bool broadcast_to_leading);

BINARY_OP_DECL(add)
BINARY_OP_DECL(sub)
BINARY_OP_DECL(mul)
BINARY_OP_DECL(div)
BINARY_OP_DECL(pow)
BINARY_OP_DECL(max)
BINARY_OP_DECL(min)


#define UNARY_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, unsigned int n, T *b); \
  template<typename T> \
  void name##_inplace(T *a, unsigned int n);

UNARY_OP_DECL(abs)
UNARY_OP_DECL(exp)
UNARY_OP_DECL(log)
UNARY_OP_DECL(relu)
UNARY_OP_DECL(relu_d)
UNARY_OP_DECL(sigmoid)
UNARY_OP_DECL(sigmoid_d)
UNARY_OP_DECL(sqrt)
UNARY_OP_DECL(tanh)
UNARY_OP_DECL(tanh_d)

}

#endif // ELEMENTWISE_HPP_
