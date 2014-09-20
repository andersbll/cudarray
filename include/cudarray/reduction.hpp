namespace cudarray {

#define REDUCE_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, unsigned int n, T *b);
REDUCE_OP_DECL(sum)


#define REDUCE_MAT_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, unsigned int m, unsigned int n, bool reduce_leading, \
            T *b);
REDUCE_MAT_OP_DECL(sum_mat)

}
