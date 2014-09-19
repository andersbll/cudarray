## CUDA-based Numpy arrays and operations


### Features
- Implements (a subset of) Numpy operations.
- Extends Numpy with specialized function for neural networks:
  - Convolution and pooling operations for convnets.
- Simple C interface based on Cython.
- Fast linear algebra module based on cuBLAS.


### TODO
- Transpose semantics
- Broadcasting semantics
- Array expressions
- Fast FFT module based on cuFFT
- Fast random number module based on cuRAND
- Unit tests!
- Proper build system


### Installation
Run

    make
    python setup.py install


#### Disable CUDA back-end
Run

    python setup.py --without-cuda install


### Requirements
 - Cython

### Influences
Thanks to the following projects for showing the way:
 - cudamat
 - PyCUDA
 - mshadow
 - pyFFTW
