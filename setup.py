#!/usr/bin/env python

import os
import fnmatch
import numpy

from setuptools import setup, find_packages, Feature
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension


def find_files(root_dir, filename_pattern):
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
      for filename in fnmatch.filter(filenames, filename_pattern):
          matches.append(os.path.join(root, filename))
    return matches


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def cuda_extensions():
    cudarray_dir = './cudarray'
    cudarray_include_dir = './include'
    cudarray_lib_dir = './build'
    include_dirs = [cudarray_include_dir, numpy.get_include()]
    cython_include_dirs = ['./cudarray/cudarray_wrap', './cudarray/cuda_wrap']
    extra_compile_args = ['-O3', '-DDEBUG', '-fPIC']
    extra_link_args = ['-fPIC']
    language = 'c++'

    cudart_ext = Extension(
        name='cudarray.cuda_wrap.cudart',
        sources=[os.path.join(cudarray_dir, 'cuda_wrap', 'cudart.pyx')],
        libraries=['cudart'],
        language='c++',
        cython_include_dirs=cython_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    cudarray_ext = Extension(
        name='cudarray.cudarray_wrap.array_data',
        sources=[os.path.join(cudarray_dir, 'cudarray_wrap', 'array_data.pyx')],
        libraries=['cudart'],
        include_dirs=include_dirs,
        library_dirs=[cudarray_lib_dir],
        cython_include_dirs=cython_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    elementwise_wrap_ext = Extension(
        name='cudarray.cudarray_wrap.elementwise',
        sources=[os.path.join(cudarray_dir, 'cudarray_wrap', 'elementwise.pyx')],
        libraries=['cudarray'],
        library_dirs=[cudarray_lib_dir],
        include_dirs=include_dirs,
        cython_include_dirs=cython_include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    reduction_wrap_ext = Extension(
        name='cudarray.cudarray_wrap.reduction',
        sources=[os.path.join(cudarray_dir, 'cudarray_wrap', 'reduction.pyx')],
        libraries=['cudarray'],
        library_dirs=[cudarray_lib_dir],
        include_dirs=include_dirs,
        cython_include_dirs=cython_include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    blas_wrap_ext = Extension(
        name='cudarray.cudarray_wrap.blas',
        sources=[os.path.join(cudarray_dir, 'cudarray_wrap', 'blas.pyx')],
        libraries=['cudarray'],
        library_dirs=[cudarray_lib_dir],
        include_dirs=include_dirs,
        cython_include_dirs=cython_include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [cudart_ext, cudarray_ext, elementwise_wrap_ext,
            reduction_wrap_ext, blas_wrap_ext]

def numpy_extensions():
    cython_srcs = [
        'cudarray/numpy_backend/nnet/conv_bc01.pyx',
        'cudarray/numpy_backend/nnet/pool_bc01.pyx',
    ]
    include_dirs = [numpy.get_include()]
    return cythonize(cython_srcs, include_path=include_dirs)


setup(
    name='cudarray',
    version='0.1',
    author='Anders Boesen Lindbo Larsen',
    author_email='abll@dtu.dk',
    description='CUDA-based Numpy array and operations',
    license='MIT',
    url='http://compute.dtu.dk/~abll',
    packages=find_packages(),
    install_requires=['numpy', 'cython'],
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
            remove=['cudarray.cudarray_wrap', 'cudarray.cuda_wrap'],
            ext_modules=cuda_extensions(),
        ),
        'numpy': Feature(
            description='Numpy back-end',
            standard=True,
            remove=['cudarray.numpy_backend'],
            ext_modules=numpy_extensions(),
        ),
    },
    cmdclass = {'build_ext': build_ext},
    zip_safe=False,
)
