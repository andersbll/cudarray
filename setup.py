#!/usr/bin/env python

import os
import numpy

from setuptools import setup, find_packages, Feature
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def cuda_extensions():
    cuda_dir = os.getenv('CUDA_PREFIX', '/appl/cuda/6.5')
    cuda_include_dir = os.path.join(cuda_dir, 'include')
    cuda_library_dir = os.path.join(cuda_dir, 'lib64')
    if not os.path.exists(cuda_library_dir):
        # Use lib if lib64 does not exist
        cuda_library_dir = os.path.join(cuda_dir, 'lib')

    library_dirs = [cuda_library_dir]
    prefix = os.getenv('INSTALL_PREFIX')
    if prefix is not None:
        library_dirs.append(os.path.join(prefix, 'lib'))

    cudarray_dir = './cudarray'
    cudarray_include_dir = './include'
    include_dirs = [cuda_include_dir, cudarray_include_dir,
                    numpy.get_include()]
    cython_include_dirs = ['./cudarray/wrap']
    extra_compile_args = ['-O3', '-fPIC', '-Wall', '-Wfatal-errors']
    libraries = ['cudart', 'cudarray']
    extra_link_args = ['-fPIC']
    language = 'c++'

    def make_extension(name):
        return Extension(
            name='cudarray.wrap.' + name,
            sources=[os.path.join(cudarray_dir, 'wrap', name + '.pyx')],
            language=language,
            include_dirs=include_dirs,
            cython_include_dirs=cython_include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args,
        )
    ext_names = ['cudart', 'array_data', 'elementwise', 'reduction', 'blas',
                 'random', 'nnet', 'nsnet']
    exts = map(make_extension, ext_names)

    if os.getenv('CUDNN_ENABLED') == '1':
        cudnn_ext = Extension(
            name='cudarray.wrap.cudnn',
            sources=[os.path.join(cudarray_dir, 'wrap', 'cudnn.pyx')],
            language=language,
            include_dirs=include_dirs,
            cython_include_dirs=cython_include_dirs,
            extra_compile_args=['-DCUDNN_ENABLED'] + extra_compile_args,
            library_dirs=library_dirs,
            libraries=libraries+['cudnn'],
            extra_link_args=extra_link_args,
        )
        exts.append(cudnn_ext)
    return exts


def numpy_extensions():
    ext_names = [
        ['nnet', 'conv_bc01'],
        ['nnet', 'pool_bc01'],
        ['nnet', 'lrnorm_bc01'],
        ['nsnet','conv_seg_bc01'],
        ['nsnet','pool_seg_bc01'],
        ['nsnet','flatten_seg_bc01']
    ]
    def make_extension(name):
        return Extension(
            name='cudarray.numpy_backend.' + name[0] +'.'+name[1],
            sources=[os.path.join('./cudarray/numpy_backend', name[0], name[1] + '.pyx')],
            include_path=[numpy.get_include()],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        )

    exts = map(make_extension, ext_names)
    return exts


setup(
    name='cudarray',
    version='0.1',
    author='Anders Boesen Lindbo Larsen',
    author_email='abll@dtu.dk',
    description='CUDA-based Numpy array and operations',
    license='MIT',
    url='http://compute.dtu.dk/~abll',
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    install_requires=['numpy', 'cython>=0.21.1'],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    features={
        'cuda': Feature(
            description='CUDA back-end',
            standard=True,
            remove=['cudarray.wrap'],
            ext_modules=cuda_extensions(),
        ),
        'numpy': Feature(
            description='Numpy back-end',
            standard=True,
            remove=['cudarray.numpy_backend'],
            ext_modules=numpy_extensions(),
        ),
    },
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
