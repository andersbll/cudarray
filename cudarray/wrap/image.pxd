cdef extern from 'cudarray/image/img2win.hpp' namespace 'cudarray':

    void img2win[T](const T *imgs, int n_imgs, int img_h, int img_w, int win_h,
        int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T *wins)

    void win2img[T](const T *wins, int n_imgs, int img_h, int img_w, int win_h,
        int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T *imgs)
