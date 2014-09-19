#ifndef LAYERS_HPP_
#define LAYERS_HPP_


void softmax(const float *x, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *out);

void softmax_bprop(const float *x, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *out);


#endif // LAYERS_HPP_
