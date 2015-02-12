
cdef extern from 'cudarray/nsnet/conv_bc01_matmul.hpp' namespace 'cudarray':

    void conv_bc01_matmul[T](const T *imgs, const T *filters, int n_imgs,
        int n_channels, int n_filters, int img_h, int img_w, int filter_h,
        int filter_w, int pad_y, int pad_x, int stride_y, int stride_x,
        T *convout)

    void conv_bc01_matmul_bprop_filters[T](const T *imgs, const T *convout_d,
        int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
        int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
        int stride_x, T *filters_d)

    void conv_bc01_matmul_bprop_imgs[T](const T *filters, const T *convout_d,
        int n_imgs, int n_channels, int n_filters, int img_h, int img_w,
        int filter_h, int filter_w, int pad_y, int pad_x, int stride_y,
        int stride_x, T *imgs_d)


cdef extern from 'cudarray/nsnet/pool_seg_b01.hpp' namespace 'cudarray':

    void max_pool_seg_b01[T](const T* imgs, int n_frag, int img_h, int img_w, 
        int n_chan, int win_h, int win_w, int pad_y, int pad_x, 
        int stride_y, int stride_x, T* out, int* mask)

    void max_pool_seg_b01_bprob[T](const T* out_d, const int* mask, int n_frag,
        int img_h, int img_w, int n_chan, int win_h, int win_w,
        int pad_y, int pad_x, int stride_y, int stride_x, T* imgs_d)


cdef extern from 'cudarray/nnet/one_hot.hpp' namespace 'cudarray':
    void one_hot_encode[T](const int *labels, int n_classes, int n, T *out)
