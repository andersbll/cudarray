#include "cudarray/common.hpp"
#include "cudarray/image/img2win.hpp"


namespace cudarray {

inline static int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

template <typename T, int group_size>
__global__ void kernel_img2win(const T *imgs, int n_threads, int n_imgs,
    int img_h, int img_w, int wins_h, int wins_w, int win_h, int win_w,
    int pad_y, int pad_x, int stride_y, int stride_x, T *wins) {
  int win_size = win_h*win_w;
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int wins_x = idx % wins_w;
    int wins_y = (idx / wins_w) % wins_h;
    // window offset
    int k = (idx / wins_w / wins_h) % win_size;
    // image idx
    int n = idx / wins_w / wins_h / win_size * group_size;

    int img_x = wins_x * stride_x - pad_x + (k % win_w);
    int img_y = wins_y * stride_y - pad_y + k / win_w;
    imgs += (n*img_h + img_y)*img_w + img_x;
    wins += ((n*win_size + k)*wins_h + wins_y)*wins_w + wins_x;
    bool valid = img_x >= 0 && img_x < img_w && img_y >= 0 && img_y < img_h;

    for (int i = 0; i < group_size; ++i) {
      if (i+n < n_imgs) {
        if (valid) {
          *wins = *imgs;
        } else {
          *wins = 0.0;
        }
      }
      wins += win_size * wins_h * wins_w;
      imgs += img_h * img_w;
    }
  }
}

template <typename T>
void img2win(const T *imgs, int n_imgs, int img_h, int img_w, int win_h,
    int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T *wins) {
  int wins_h = (img_h + 2*pad_y - win_h) / stride_y + 1;
  int wins_w = (img_w + 2*pad_x - win_w) / stride_x + 1;
  const int group_size = 32;
  int n_threads = ceil_div(n_imgs, group_size)*win_h*win_w*wins_h*wins_w;
  kernel_img2win<T, group_size>
      <<<cuda_blocks(n_threads), kNumBlockThreads>>>(
      imgs, n_threads, n_imgs, img_h, img_w, wins_h, wins_w, win_h, win_w,
      pad_y, pad_x, stride_y, stride_x, wins
  );
  CUDA_KERNEL_CHECK;
}
template void img2win(const float *imgs, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *wins);



template <typename T>
__global__ void kernel_win2img(const T* wins, int n_threads, int n_imgs,
    int img_h, int img_w, int wins_h, int wins_w, int win_h, int win_w,
    int pad_y, int pad_x, int stride_y, int stride_x, T *imgs) {
  CUDA_GRID_STRIDE_LOOP(idx, n_threads) {
    int img_x = idx % img_w + pad_x;
    int img_y = (idx / img_w) % img_h + pad_y;
    int n = idx / img_w / img_h;

    int wins_x_start = (img_x < win_w) ? 0 : (img_x - win_w) / stride_x + 1;
    int wins_x_end = min(img_x / stride_x + 1, wins_w);
    int wins_y_start = (img_y < win_h) ? 0 : (img_y - win_h) / stride_y + 1;
    int wins_y_end = min(img_y / stride_y + 1, wins_h);

    int wins_y_offset = (1 - stride_y * win_w * wins_h) * wins_w;
    int wins_x_offset = (1 - stride_x * wins_h * wins_w);

    wins += (n * win_h * win_w + img_y * win_w + img_x) * wins_h * wins_w;
    T sum = 0;
    for (int wins_y = wins_y_start; wins_y < wins_y_end; ++wins_y) {
      for (int wins_x = wins_x_start; wins_x < wins_x_end; ++wins_x) {
        sum += wins[wins_y * wins_y_offset + wins_x * wins_x_offset];
      }
    }
    imgs[idx] = sum;
  }
}

template <typename T>
void win2img(const T *wins, int n_imgs, int img_h, int img_w, int win_h,
    int win_w, int pad_y, int pad_x, int stride_y, int stride_x, T *imgs) {
  int wins_h = (img_h + 2*pad_y - win_h) / stride_y + 1;
  int wins_w = (img_w + 2*pad_x - win_w) / stride_x + 1;
  int n_threads = n_imgs * img_h * img_w;
  kernel_win2img<<<cuda_blocks(n_threads), kNumBlockThreads>>>(
      wins, n_threads, n_imgs, img_h, img_w, wins_h, wins_w, win_h, win_w,
      pad_y, pad_x, stride_y, stride_x, imgs);
  CUDA_KERNEL_CHECK;
}

template void win2img(const float *wins, int n_imgs, int img_h, int img_w,
    int win_h, int win_w, int pad_y, int pad_x, int stride_y, int stride_x,
    float *imgs);
}
