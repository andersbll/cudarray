#ifndef IMG2WIN_HPP_
#define IMG2WIN_HPP_

namespace cudarray {

template <typename T>
void img2win(const T *imgs, int n_imgs, int img_h, int img_w, int win_h,
    int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T *wins);

template <typename T>
void win2img(const T *wins, int n_imgs, int img_h, int img_w, int win_h,
    int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T *imgs);

}

#endif // IMG2WIN_HPP_
