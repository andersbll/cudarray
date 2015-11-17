cdef extern from 'cudarray/image/rescale.hpp' namespace 'cudarray':
    enum SampleMethod:
        BILINEAR_SAMPLING
        NEAREST_SAMPLING
        PERFORATED_SAMPLING

    void rescale[T](const T *imgs, float factor, SampleMethod method,
                    int n_imgs, int img_h, int img_w, T *imgs_scaled)

