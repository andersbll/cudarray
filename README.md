## CUDA-based Numpy arrays and operations


### Features
- Implements (a subset of) Numpy operations.
- Fast linear algebra module based on cuBLAS.
- Extends Numpy with specialized functions for neural networks:
  - Convolution and pooling operations for convnets.
- (somewhat) Simple C/CUDA wrapper based on Cython.


#### Limitations compared to Numpy
- Can only reduce (`sum`, `max`, etc.) along either leading or trailing axes.


### TODO
- Transpose semantics
- Broadcasting semantics
- Array expressions
- Fast FFT module based on cuFFT
- Fast random number module based on cuRAND
- Unit tests!
- Proper build system


### Installation
Install with CUDA back-end:

    make
    python setup.py install


Install without CUDA back-end:

    python setup.py --without-cuda install


### Requirements
 - Cython

### Influences
Thanks to the following projects for showing the way:
 - cudamat
 - PyCUDA
 - mshadow
 - pyFFTW
