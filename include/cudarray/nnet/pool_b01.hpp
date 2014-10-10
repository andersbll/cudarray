#ifndef POOL_B01_HPP_
#define POOL_B01_HPP_

namespace cudarray {

template <typename T>
void max_pool_b01(const T* imgs, int n_imgs, int img_h, int img_w, int win_h,
    int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T* poolout,
    int* mask);

template <typename T>
void max_pool_b01_bprob(const T* poolout_d, const int* mask, int n_imgs,
    int img_h, int img_w, int win_h, int win_w, int pad_y, int pad_x,
    int stride_y, int stride_x, T* imgs_d);

}

#endif  // POOL_B01_HPP_
