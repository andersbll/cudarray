namespace cudarray {

//#define REDUCE_OP_DECL(name) \
//  template<typename T> \
//  void sum(const T *a, unsigned int n, T *b);
//REDUCE_OP_DECL(sum)
//REDUCE_OP_DECL(mean)
//REDUCE_OP_DECL(max)
//REDUCE_OP_DECL(min)


#define BATCH_REDUCE_OP_DECL(name) \
  template<typename T> \
  void name(const T *a, unsigned int m, unsigned int n, bool reduce_leading, \
            T *b);
BATCH_REDUCE_OP_DECL(sum_batched)

}
