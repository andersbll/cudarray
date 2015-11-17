#ifndef RESCALE_HPP_
#define RESCALE_HPP_

namespace cudarray {

enum SampleMethod {
  BILINEAR_SAMPLING, NEAREST_SAMPLING, PERFORATED_SAMPLING,
};

template <typename T>
void rescale(const T *imgs, float factor, SampleMethod method, int n_imgs,
             int img_h, int img_w, T *imgs_scaled);

}

#endif // RESCALE_HPP_
