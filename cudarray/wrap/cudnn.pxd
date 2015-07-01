cdef extern from "cudarray/nnet/cudnn.hpp" namespace 'cudarray':
    enum PoolMode:
        POOL_AVG
        POOL_MAX

    cdef cppclass PoolBC01CuDNN[T]:
        PoolBC01CuDNN(int n_img_dims, int *win_shape, int *padding,
                      int *strides, PoolMode pool_mode)

        void fprop(const T *imgs, int *imgs_shape, T *poolout)

        void bprop(const T *imgs, const T* poolout, const T *poolout_d,
                   T *imgs_d)


    cdef cppclass ConvBC01CuDNN[T]:
      ConvBC01CuDNN(int pad_y, int pad_x, int stride_y, int stride_x)

      void fprop(const T *imgs, const T *filters, int n_imgs, int n_channels,
          int n_filters, int img_h, int img_w, int filter_h, int filter_w,
          T *convout)

      void bprop(const T *imgs, const T *filters, const T *convout_d,
                 T *imgs_d, T *filters_d)
