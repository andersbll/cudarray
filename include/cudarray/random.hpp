#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>
#include <curand.h>

const char* curandGetErrorString(curandStatus_t error);

#define CURAND_CHECK(expr) \
  { \
    curandStatus_t status = expr; \
    if (status != CURAND_STATUS_SUCCESS) { \
        throw std::runtime_error(curandGetErrorString(status)); \
    } \
  }

namespace cudarray {

void seed(unsigned long long val);

template <typename T>
void random_normal(T *a, T mu, T sigma, unsigned int n);

template <typename T>
void random_uniform(T *a, T low, T high, unsigned int n);

/*
  Singleton class to handle cuRAND resources.
*/
class CURAND {
public:
  inline static CURAND &instance() {
    static CURAND instance_;
    return instance_;
  }

  inline static curandGenerator_t &generator() {
    return instance().generator_;
  }

private:
  curandGenerator_t generator_;
  CURAND() {
    CURAND_CHECK(curandCreateGenerator(&generator_,
                                       CURAND_RNG_PSEUDO_DEFAULT));
    std::srand(std::time(NULL)+getpid());
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, std::rand()));
  }
  ~CURAND() {
  }
  CURAND(CURAND const&);
  void operator=(CURAND const&);
};

}

#endif // RANDOM_HPP_
