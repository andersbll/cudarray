#ifndef ONE_HOT_HPP_
#define ONE_HOT_HPP_

namespace cudarray {

template <typename T>
void one_hot_encode(const int *labels, int n_classes, int n, T *out);

}

#endif  // ONE_HOT_HPP_
