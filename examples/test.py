#!/usr/bin/env python

import os
import numpy as np

os.environ['CUDARRAY_BACKEND'] = 'cuda'
#os.environ['CUDARRAY_BACKEND'] = 'numpy'
import cudarray as ca


def test_dot():
    a = np.random.normal(size=(5,5))
    b = np.random.normal(size=(5,5))
    c_np = np.dot(a,b)

    a = ca.array(a)
    b = ca.array(b)

    c_ca = ca.dot(a, b)
    print(np.allclose(c_np, np.array(c_ca)))

    c_ca = ca.zeros_like(a)
    ca.dot(a, b, c_ca)
    print(np.allclose(c_np, np.array(c_ca)))


def test_multiply():
    a = np.random.normal(size=(5,5))
    a_ca = ca.array(a)

    c_np = np.multiply(a, 3)
    c_ca = ca.multiply(a_ca, 3)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.multiply(a, 3, a)
    c_ca = ca.multiply(a_ca, 3, a_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a = np.random.normal(size=(5,5))
    a_ca = ca.array(a)
    b = np.random.normal(size=(5,5))
    b_ca = ca.array(b)

    c_np = np.multiply(a, b, a)
    c_ca = ca.multiply(a_ca, b_ca, a_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.multiply(a, b)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


def test_sum():
    a_np = np.random.normal(size=(5,5))
    a_ca = ca.array(a_np)

    s_np = np.sum(a_np, 0)
    s_ca = ca.sum(a_ca, 0)
    print(np.allclose(s_np, np.array(s_ca)))

    s_np = np.sum(a_np, 1)
    s_ca = ca.sum(a_ca, 1)
    print(np.allclose(s_np, np.array(s_ca)))

    a_np = np.random.normal(size=(5, 5, 10))
    a_ca = ca.array(a_np)

    s_np = np.sum(a_np, 0)
    s_ca = ca.sum(a_ca, 0)
    print(np.allclose(s_np, np.array(s_ca)))

    s_np = np.sum(a_np, 2)
    s_ca = ca.sum(a_ca, 2)
    print(np.allclose(s_np, np.array(s_ca)))


def run():
#    test_dot()
    test_multiply()
    test_sum()
    return


if __name__ == '__main__':
    run()
