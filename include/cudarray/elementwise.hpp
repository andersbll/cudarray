#ifndef ELEMENTWISE_HPP_
#define ELEMENTWISE_HPP_

#include <cudarray/common.hpp>


namespace cudarray {

enum BinaryOp {
  ADD_OP, DIV_OP, MAX_B_OP, MIN_B_OP, MUL_OP, POW_OP, SUB_OP,
};

template<typename Ta, typename Tb, typename Tc>
void binary(BinaryOp op, const Ta *a, const Tb *b, unsigned int n, Tc *c);

template<typename Ta, typename Talpha, typename Tb>
void binary_scalar(BinaryOp op, const Ta *a, Talpha alpha, unsigned int n,
                   Tb *b);

template<typename Ta, typename Tb, typename Tc>
void binary_broadcast(BinaryOp op, const Ta *a, const Tb *b, unsigned int m,
                      unsigned int n, bool broadcast_to_leading, Tc *c);


enum BinaryCmpOp {
  EQ_OP, GT_OP, GT_EQ_OP, LT_OP, LT_EQ_OP, NEQ_OP,
};

template<typename Ta, typename Tb>
void binary_cmp(BinaryCmpOp op, const Ta *a, const Tb *b, unsigned int n,
                bool_t *c);

template<typename Ta, typename Talpha>
void binary_cmp_scalar(BinaryCmpOp op, const Ta *a, Talpha alpha,
                       unsigned int n, bool_t *b);

template<typename Ta, typename Tb>
void binary_cmp_broadcast(BinaryCmpOp op, const Ta *a, const Tb *b,
    unsigned int m, unsigned int n, bool broadcast_to_leading, bool_t *c);


enum UnaryOp {
  ABS_OP, COS_OP, EXP_OP, LOG_OP, NEG_OP, SIN_OP, SQRT_OP, TANH_OP,
  RELU_OP, RELU_D_OP, SIGMOID_OP, SIGMOID_D_OP, TANH_D_OP,
};

template<typename T>
void unary(UnaryOp op, const T *a, unsigned int n, T *b);

template<typename T>
void clip(const T *a, T a_min, T a_max, unsigned int n, T *b);

}

#endif // ELEMENTWISE_HPP_
