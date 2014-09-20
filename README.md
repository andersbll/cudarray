## CUDA-based Numpy arrays and operations


### Features
- Drop-in replacement for Numpy (limitations apply).
- Fast linear algebra module based on cuBLAS.
- (somewhat) Simple C++/CUDA wrapper based on Cython.
- Extends Numpy with specialized functions for neural networks.


### Limitations
- Element-wise operations (`+`, `*`, etc.) can broadcast only along either leading or trailing axes.
- Reduction operations (`sum`, `max`, etc.) is supported on either leading or trailing axes.


### Installation
Install with CUDA back-end:

    make
    python setup.py install

Install without CUDA back-end:

    python setup.py --without-cuda install


### TODO
- Transpose semantics
- Array expressions
- Fast FFT module based on cuFFT
- Fast random number module based on cuRAND
- Unit tests!
- Proper build system


### Influences
Thanks to the following projects for showing the way.
 - cudamat
 - PyCUDA
 - mshadow
 - pyFFTW
