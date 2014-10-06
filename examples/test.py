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


    a_np = np.random.normal(size=(5))
    b_np = np.random.normal(size=(5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np, b_np)
    c_ca = ca.dot(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.random.normal(size=(5, 5))
    b_np = np.random.normal(size=(5, 5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np.T, b_np)
    c_ca = ca.dot(a_ca.T, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.random.normal(size=(3, 4))
    b_np = np.random.normal(size=(5, 4))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np, b_np.T)
    c_ca = ca.dot(a_ca, b_ca.T)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(4, 3))
    b_np = np.random.normal(size=(4, 5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np.T, b_np)
    c_ca = ca.dot(a_ca.T, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(4, 3))
    b_np = np.random.normal(size=(5, 4))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np.T, b_np.T)
    c_ca = ca.dot(a_ca.T, b_ca.T)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.random.normal(size=(4))
    b_np = np.random.normal(size=(4, 5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np, b_np)
    c_ca = ca.dot(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(4, 5))
    b_np = np.random.normal(size=(5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np, b_np)
    c_ca = ca.dot(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(4))
    b_np = np.random.normal(size=(5, 4))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np, b_np.T)
    c_ca = ca.dot(a_ca, b_ca.T)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(5, 4))
    b_np = np.random.normal(size=(5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.dot(a_np.T, b_np)
    c_ca = ca.dot(a_ca.T, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

def test_multiply():
    a_np = np.ones((5,5))
    b_np = np.arange(5)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_ca = ca.multiply(a_ca, b_ca, a_ca)
    c_np = np.multiply(a_np, b_np, a_np)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.ones((3,3))
    b_np = np.arange(3)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.ones((3,3))
    b_np = np.arange(3).reshape(1,3)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.ones((3,3))
    b_np = np.arange(3).reshape(3,1)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)

    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.ones((3, 3, 4))
    b_np = np.arange(3).reshape(3,1,1)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.ones((3, 3, 4))
    b_np = np.arange(4)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.arange(3)
    b_np = np.ones((3,3))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


    a_np = np.arange(4)
    b_np = np.ones((3, 3, 4))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

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

def test_binary():
    a_np = np.random.normal(size=(5,5))
    b_np = np.random.normal(size=(5,5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)

    c_np = np.add(a_np, b_np)
    c_ca = ca.add(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    np.add(a_np, b_np, a_np)
    ca.add(a_ca, b_ca, a_ca)
    print(np.allclose(a_np, np.array(a_ca)))


    np.multiply(a_np, b_np, a_np)
    ca.multiply(a_ca, b_ca, a_ca)
    print(np.allclose(a_np, np.array(a_ca)))

    a_np = np.random.normal(size=(5,5))
    b_np = np.random.normal(size=(5,5)) > 0
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


def test_binary_cmp():
    a_np = np.random.normal(size=(5,5))
    b_np = np.random.normal(size=(5,5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)

    c_np = np.greater(a_np, b_np)
    c_ca = ca.greater(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.greater(a_np, 0.1)
    c_ca = ca.greater(a_ca, 0.1)
    print(np.allclose(c_np, np.array(c_ca)))


    c_np = np.less(a_np, 0.1)
    c_ca = ca.less(a_ca, 0.1)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = 0.1 < a_np
    c_ca = 0.1 < a_ca
    print(np.allclose(c_np, np.array(c_ca)))


def test_sum():
    a_np = np.random.normal(size=(5,5))
    a_ca = ca.array(a_np)

    s_np = np.sum(a_np)
    s_ca = ca.sum(a_ca)
    print(np.allclose(s_np, np.array(s_ca)))

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


def test_random():
    a_np = np.random.normal(size=(1000,1000), loc=3, scale=10)
    a_ca = ca.random.normal(size=(1000,1000), loc=3, scale=10)
    a_ca = np.array(a_ca)
    print(np.mean(a_np), np.mean(a_ca))
    print(np.std(a_np), np.std(a_ca))

    a_np = np.random.uniform(size=(1000,1000), low=0, high=1)
    a_ca = ca.random.uniform(size=(1000,1000), low=0, high=1)
    a_ca = np.array(a_ca)
    print(np.mean(a_np), np.mean(a_ca))
    print(np.std(a_np), np.std(a_ca))

    a_np = np.random.uniform(size=(1000,1000), low=-10, high=30)
    a_ca = ca.random.uniform(size=(1000,1000), low=-10, high=30)
    a_ca = np.array(a_ca)
    print(np.mean(a_np), np.mean(a_ca))
    print(np.std(a_np), np.std(a_ca))




def run():
#    test_dot()
    test_multiply()
    test_binary()
    test_binary_cmp()
#    test_sum()
#    test_random()
    return


if __name__ == '__main__':
    run()
