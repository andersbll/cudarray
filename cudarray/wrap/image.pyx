cimport numpy as np
cimport image
from .array_data cimport (ArrayData, float_ptr, int_ptr, is_int, is_float)


sample_methods = {
    'bilinear': BILINEAR_SAMPLING,
    'nearest': NEAREST_SAMPLING,
    'perforated': PERFORATED_SAMPLING,
}


def _rescale(ArrayData imgs, float factor, SampleMethod method, int n_imgs,
             int img_h, int img_w, ArrayData out):
    if is_float(imgs):
        image.rescale(float_ptr(imgs), factor, method, n_imgs, img_h, img_w,
                    float_ptr(out))
    else:
        raise ValueError('type (%s) not implemented' % str(out.dtype))

