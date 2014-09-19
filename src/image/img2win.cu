#include "cudarray/common.hpp"
#include "cudarray/image/img2win.hpp"

inline static int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

template <int group_size>
__global__ void img2win_kernel(const float *imgs, int n_threads, int n_imgs, int img_h,
    int img_w, int out_h, int out_w, int win_h, int win_w, int pad_y,
    int pad_x, int stride_y, int stride_x, float *out) {
  int win_size = win_h*win_w;
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int out_x = idx % out_w;
    int out_y = (idx / out_w) % out_h;
    // window offset
    int k = (idx / out_w / out_h) % win_size;
    // image index
    int n = idx / out_w / out_h / win_size * group_size;

    int img_x = out_x * stride_x - pad_x + (k % win_w);
    int img_y = out_y * stride_y - pad_y + k / win_w;
    const float *img_ = imgs + (n*img_h + img_y)*img_w + img_x;
    float *out_ = out + ((n*win_size + k)*out_h + out_y)*out_w + out_x;
    float valid = img_x >= 0 && img_x < img_w && img_y >= 0 && img_y < img_h ? 1.0 : 0.0;

    #pragma unroll
    for (int i = 0; i < group_size; ++i) {
      // XXX: is this faster?
      // *out_ = i+n < n_imgs ? *img_ : 0.0;
      if (i+n < n_imgs) {
        *out_ = *img_ * valid;
      }
      out_ += win_size * out_h * out_w;
      img_ += img_h * img_w;
    }
  }
}


void img2win(const float *imgs, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *out) {
  int out_h = (img_h + 2*pad_y - win_h) / stride_y + 1;
  int out_w = (img_w + 2*pad_x - win_w) / stride_x + 1;
  int group_size = 32;
  int n_threads = ceil_div(n_imgs, group_size) * win_h * win_w * out_h * out_w;
//  std::cout << n_threads << "  " << n_imgs << "  " << win_h << "  " << win_w << "  " << pad_y << "  " << pad_x << "  " << stride_y << "  " << stride_x << std::endl;
//  std::cout << n_imgs << "  " << group_size << "  " << ceil_div(n_imgs, group_size) << "  " << n_threads << "  " << CUDA_BLOCKS(n_threads) << "  " << CUDA_NUM_THREADS << std::endl;
  img2win_kernel<32>
      <<<CUDA_BLOCKS(n_threads), CUDA_NUM_THREADS>>>(
      imgs, n_threads, n_imgs, img_h, img_w, out_h, out_w, win_h, win_w, pad_y, pad_x,
      stride_y, stride_x, out
  );
  CUDA_DEBUG_SYNC("img2win failed");
}




/*
  imgs: nyx , n = n_imgs, y = img_h, x = img_w
  out: nkyx , k = win_w*win_h, y = out_h, x = out_w
  n_threads = n/g * k * out_h * out_w, g = group_size
*/
template <int group_size>
__global__ void img2invwin_kernel(const float *imgs, int n_threads, int n_imgs, int img_h,
    int img_w, int out_h, int out_w, int win_h, int win_w, int pad_y,
    int pad_x, int stride_y, int stride_x, float *out) {
  int win_size = win_h*win_w;
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int out_x = idx % out_w;
    int out_y = (idx / out_w) % out_h;
    // window offset
    int k = (idx / out_w / out_h) % win_size;
    // image index
    int n = idx / out_w / out_h / win_size * group_size;

    int img_x = (out_x - (k % win_w) + pad_x) / stride_x;
    int img_y = (out_y - (k / win_w) + pad_y) / stride_y;
    float valid = img_x % stride_x == 0 && img_y % stride_y == 0 ? 1.0 : 0.0;
    img_x /= stride_x;
    img_y /= stride_y;

    const float *img_ = imgs + (n*img_h + img_y)*img_w + img_x;
    float *out_ = out + ((n*win_size + k)*out_h + out_y)*out_w + out_x;
    valid *= img_x >= 0 && img_x < img_w && img_y >= 0 && img_y < img_h ? 1.0 : 0.0;

    #pragma unroll
    for (int i = 0; i < group_size; ++i) {
      if (i+n < n_imgs) {
        *out_ = *img_ * valid;
      }
      out_ += win_size * out_h * out_w;
      img_ += img_h * img_w;
    }
  }
}


void img2invwin(const float *imgs, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *out) {
  int out_h = (img_h - 1) * stride_y - 2*pad_y + win_h;
  int out_w = (img_w - 1) * stride_x - 2*pad_x + win_w;
  int group_size = 32;
  int n_threads = ceil_div(n_imgs, group_size) * win_h * win_w * out_h * out_w;
//  std::cout << n_threads << "  " << n_imgs << "  " << win_h << "  " << win_w << "  " << pad_y << "  " << pad_x << "  " << stride_y << "  " << stride_x << std::endl;
//  std::cout << n_imgs << "  " << group_size << "  " << ceil_div(n_imgs, group_size) << "  " << n_threads << "  " << CUDA_BLOCKS(n_threads) << "  " << CUDA_NUM_THREADS << std::endl;
  img2invwin_kernel<32>
      <<<CUDA_BLOCKS(n_threads), CUDA_NUM_THREADS>>>(
      imgs, n_threads, n_imgs, img_h, img_w, out_h, out_w, win_h, win_w, pad_y, pad_x,
      stride_y, stride_x, out
  );
  CUDA_DEBUG_SYNC("img2win failed");
}
