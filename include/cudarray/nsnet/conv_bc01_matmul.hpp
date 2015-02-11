#ifndef CONV_BC01_MATMUL_HPP_
#define CONV_BC01_MATMUL_HPP_

namespace cudarray {

template <typename T>
void conv_bc01_matmul(const T *imgs, const T *filters, int n_imgs,
    int n_channels, int n_filters, int img_h, int img_w, int filter_h,
    int filter_w, int pad_y, int pad_x, int stride_y, int stride_x,
    T *convout);

template <typename T>
void conv_bc01_matmul_bprop_imgs(const T *filters, const T *convout_d,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, T *imgs_d);

template <typename T>
void conv_bc01_matmul_bprop_filters(const T *imgs, const T *convout_d,
    int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
    int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
    int stride_x, T *filters_d);

}

#endif // CONV_BC01_MATMUL_HPP_
