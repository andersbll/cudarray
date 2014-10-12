#ifndef REDUCTION_HPP_
#define REDUCTION_HPP_

namespace cudarray {

enum ReduceOp {
  MAX_OP, MEAN_OP, MIN_OP, SUM_OP
};

enum ReduceToIntOp {
  ARGMAX_OP, ARGMIN_OP
};

template<typename T>
void reduce(ReduceOp op, const T *a, unsigned int n, T *b);

template<typename T>
void reduce_mat(ReduceOp op, const T *a, unsigned int m, unsigned int n,
                bool reduce_leading, T *b);

template<typename T>
void reduce_to_int(ReduceToIntOp op, const T *a, unsigned int n, int *b);

template<typename T>
void reduce_mat_to_int(ReduceToIntOp op, const T *a, unsigned int m,
                       unsigned int n, bool reduce_leading, int *b);

}

#endif // REDUCTION_HPP_
