cdef extern from "cudarray/nnet/cudnn.hpp" namespace 'cudarray':
    cdef cppclass PoolBC01CuDNN[T]:
        PoolBC01CuDNN(int win_h, int win_w, int pad_y, int pad_x, int stride_y,
                      int stride_x)

        void fprop(const T *imgs, int n_imgs, int n_channels, int img_h,
                  int img_w, T *poolout)

        void bprop(const T *imgs, const T* poolout, const T *poolout_d,
                   T *imgs_d)


    cdef cppclass ConvBC01CuDNN[T]:
      ConvBC01CuDNN(int pad_y, int pad_x, int stride_y, int stride_x)

      void fprop(const T *imgs, const T *filters, int n_imgs, int n_channels,
          int n_filters, int img_h, int img_w, int filter_h, int filter_w,
          T *convout)

      void bprop(const T *imgs, const T *filters, const T *convout_d,
                 T *imgs_d, T *filters_d)
