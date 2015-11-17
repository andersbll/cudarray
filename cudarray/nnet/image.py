import numpy as np
import cudarray as ca
from ..wrap import image


def rescale(imgs, factor, method, out=None):
    img_h, img_w = imgs.shape[-2:]
    batch_shape = imgs.shape[:-2]
    n_imgs = np.prod(batch_shape)
    if factor > 1:
        scaled_h = int(np.floor(img_h * factor))
        scaled_w = int(np.floor(img_w * factor))
    else:
        scaled_h = int(np.ceil(img_h * factor))
        scaled_w = int(np.ceil(img_w * factor))
    out_shape = batch_shape + (scaled_h, scaled_w)
    if out is None:
        out = ca.empty(out_shape, dtype=imgs.dtype)
    else:
        if out.shape != out_shape:
            raise ValueError('shape mismatch')
    method = image.sample_methods[method]
    image._rescale(imgs._data, factor, method, n_imgs, img_h, img_w, out._data)
    return out
