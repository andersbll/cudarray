// TODO: implement the rest from http://docs.nvidia.com/cuda/cuda-math-api

namespace cudarray {

#define ELEMENTWISE_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, const T *b, int n, T *c);
ELEMENTWISE_OP_DECL(add)
ELEMENTWISE_OP_DECL(sub)
ELEMENTWISE_OP_DECL(mul)
ELEMENTWISE_OP_DECL(div)


#define ELEMENTWISE_BROADCAST_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, const T *b, int m, int n, bool broadcast_to_leading, \
            T *c);
ELEMENTWISE_BROADCAST_OP_DECL(add_broadcast)
ELEMENTWISE_BROADCAST_OP_DECL(mul_broadcast)


#define ELEMENTWISE_INPLACE_OP_DECL(name) \
  template<typename T> \
  void name(T *x, const T *y, int n);
ELEMENTWISE_INPLACE_OP_DECL(add_inplace)
ELEMENTWISE_INPLACE_OP_DECL(sub_inplace)
ELEMENTWISE_INPLACE_OP_DECL(mul_inplace)
ELEMENTWISE_INPLACE_OP_DECL(div_inplace)


#define SCALAR_OP_DECL(name) \
  template<typename T> \
  void name(const T *x, T alpha, int n, T *y);
SCALAR_OP_DECL(add_scalar)
SCALAR_OP_DECL(sub_scalar)
SCALAR_OP_DECL(mul_scalar)
SCALAR_OP_DECL(div_scalar)


#define SCALAR_INPLACE_OP_DECL(name) \
  template<typename T> \
  void name(T *x, T alpha, int n);
SCALAR_INPLACE_OP_DECL(add_scalar_inplace)
SCALAR_INPLACE_OP_DECL(sub_scalar_inplace)
SCALAR_INPLACE_OP_DECL(mul_scalar_inplace)
SCALAR_INPLACE_OP_DECL(div_scalar_inplace)


#define UNARY_OP_DECL(name) \
  template<typename T> \
  void name(const T *x, int n, T *y);
UNARY_OP_DECL(abs)
UNARY_OP_DECL(exp)
UNARY_OP_DECL(sqrt)
UNARY_OP_DECL(log)
UNARY_OP_DECL(tanh)


#define UNARY_ARG_OP_DECL(name) \
  template<typename T> \
  void name(const T *x, T arg, int n, T *y);
UNARY_ARG_OP_DECL(pow)
UNARY_ARG_OP_DECL(max)
UNARY_ARG_OP_DECL(min)

}
