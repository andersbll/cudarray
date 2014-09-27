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
##### With CUDA back-end
First, build libcudarray:

    make

Install libcudarray to system folders. Beforehand, you should set the system path where Python looks for libraries. For Anaconda Python distributions, something similar to the following should work.

    export INSTALL_PREFIX=/path/to/anaconda
    make install

Alternatively, you can skip the previous installation step by adding the build dir to your library search path.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudarray/build

Install the cudarray package:

    python setup.py install


##### Without CUDA back-end
Install the cudarray package:

    python setup.py --without-cuda install


### TODO
- Transpose semantics
- Array expressions
- Fast FFT module based on cuFFT
- Unit tests!


### Influences
Thanks to the following projects for showing the way.
 - cudamat
 - PyCUDA
 - mshadow
