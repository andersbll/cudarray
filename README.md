## CUDA-based NumPy

CUDArray is a CUDA-accelerated subset of the [NumPy](http://www.numpy.org/) library.
The goal of CUDArray is to combine the easy of development from the NumPy with the computational power of Nvidia GPUs in a lightweight and extensible framework.

CUDArray currently imposes many limitations in order to span a manageable subset of the NumPy library.
Nonetheless, it supports a neural network pipeline as demonstrated in the project [deeppy](http://github.com/andersbll/deeppy/).


### Features
- Drop-in replacement for NumPy (limitations apply).
- Fast array operations based on cuBLAS, cuRAND and cuDNN.
- (somewhat) Simple C++/CUDA wrapper based on Cython.
- Extends NumPy with specialized functions for neural networks.
- CPU fall-back when CUDA is not available 


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


### Documentation
Please consult the [technical report](http://www2.compute.dtu.dk/~abll/pubs/larsen2014cudarray.pdf) for now.
Proper documentation is on the TODO list.


### Contact
Feel free to report an [issue](http://github.com/andersbll/cudarray/issues) for feature requests and bug reports.

For a more informal chat, visit #cudarray on the [freenode](http://freenode.net/) IRC network.


### Citation
If you use CUDArray for research, please cite the technical report:
```
@techreport{larsen2014cudarray,
  author = "Larsen, Anders Boesen Lindbo",
  title = "{CUDArray}: {CUDA}-based {NumPy}",
  institution = "Department of Applied Mathematics and Computer Science, Technical University of Denmark",
  year = "2014",
  number = "DTU Compute 2014-21",
}
```


### TODO
- Proper transpose support,
- Copy from NumPy array to existing CUDArray array.
- FFT module based on cuFFT,
- Unit tests!
- Add documentation to wiki.
- Windows/OS X support.


### Influences
Thanks to the following projects for inspiration.
 - [cudamat](http://github.com/cudamat/cudamat)
 - [PyCUDA](http://mathema.tician.de/software/pycuda/)
 - [mshadow](http://github.com/tqchen/mshadow/)
 - [Caffe](http://caffe.berkeleyvision.org/)
 - [CUDPP](http://cudpp.github.io/)
