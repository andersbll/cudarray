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
- CPU fall-back when CUDA is not available.


### Installation
##### With CUDA back-end
First, you should consider specifying the following environment variables.
 - `INSTALL_PREFIX` (default: `/usr/local`). libcudarray is installed under `$INSTALL_PREFIX/lib`. For the Anaconda Python distrubtion this should be `/path/to/anaconda`.
 - `CUDA_DIR` (default: `/usr/local/cuda`). Path to the CUDA SDK organized in `bin/`, `lib/`, `include/` folders.
 - `CUDNN_ENABLED`. Set `CUDNN_ENABLED` to `1` to include cuDNN operations in `libcudarray`.

Then build and install libcudarray with

    make
    make install

Finally, install the cudarray Python package:

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

    @techreport{larsen2014cudarray,
      author = "Larsen, Anders Boesen Lindbo",
      title = "{CUDArray}: {CUDA}-based {NumPy}",
      institution = "Department of Applied Mathematics and Computer Science, Technical University of Denmark",
      year = "2014",
      number = "DTU Compute 2014-21",
    }


### TODO
- Proper transpose support,
- Add functionality for copying from NumPy array to existing CUDArray array.
- FFT module based on cuFFT.
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
