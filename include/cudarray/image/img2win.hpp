#ifndef IMG2WIN_HPP_
#define IMG2WIN_HPP_


void img2win(const float *imgs, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *out);

void img2invwin(const float *imgs, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *out);

#endif // IMG2WIN_HPP_
