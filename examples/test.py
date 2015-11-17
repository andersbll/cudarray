#!/usr/bin/env python

import os
import numpy as np

os.environ['CUDARRAY_BACKEND'] = 'cuda'
import cudarray as ca


def test_dot():
    a = np.random.normal(size=(5, 5))
    b = np.random.normal(size=(5, 5))
    c_np = np.dot(a, b)

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


def test_batch_dot():
    batch_size = 10
    a_np = np.random.normal(size=(batch_size, 5, 5))
    b_np = np.random.normal(size=(batch_size, 5, 5))
    c_np = np.empty_like(a_np)
    for i in range(batch_size):
        c_np[i] = np.dot(a_np[i], b_np[i])

    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_ca = ca.array(c_np)
    bdot = ca.batch.Dot(a_ca, b_ca, c_ca)
    bdot.perform()
    print(np.allclose(c_np, np.array(c_ca)))


def test_multiply():
    a_np = np.ones((5, 5))
    b_np = np.arange(5)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_ca = ca.multiply(a_ca, b_ca, a_ca)
    c_np = np.multiply(a_np, b_np, a_np)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.ones((3, 3))
    b_np = np.arange(3)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.ones((3, 3))
    b_np = np.arange(3).reshape(1, 3)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.ones((3, 3))
    b_np = np.arange(3).reshape(3, 1)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)

    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.ones((3, 3, 4))
    b_np = np.arange(3).reshape(3, 1, 1)
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
    b_np = np.ones((3, 3))
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

    a_np = np.ones((2, 7, 3, 5, 6))
    b_np = np.arange(3).reshape(1, 1, 3, 1, 1)
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.ones((3, 3, 4))
    b_np = np.ones((3, 1, 4))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a = np.random.normal(size=(5, 5))
    a_ca = ca.array(a)
    c_np = np.multiply(a, 3)
    c_ca = ca.multiply(a_ca, 3)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.multiply(a, 3, a)
    c_ca = ca.multiply(a_ca, 3, a_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a = np.random.normal(size=(5, 5))
    a_ca = ca.array(a)
    b = np.random.normal(size=(5, 5))
    b_ca = ca.array(b)

    c_np = np.multiply(a, b, a)
    c_ca = ca.multiply(a_ca, b_ca, a_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.multiply(a, b)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


def test_binary():
    a_np = np.random.normal(size=(5, 5))
    b_np = np.random.normal(size=(5, 5))
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

    a_np = np.random.normal(size=(5, 5))
    b_np = np.random.normal(size=(5, 5)) > 0
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal()
    b_np = np.random.normal(size=(5, 5))
    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.multiply(a_np, b_np)
    c_ca = ca.multiply(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.divide(a_np, b_np)
    c_ca = ca.divide(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_ca = ca.array(a_np)
    b_ca = ca.array(b_np)
    c_np = np.subtract(a_np, b_np)
    c_ca = ca.subtract(a_ca, b_ca)
    print(np.allclose(c_np, np.array(c_ca)))


def test_binary_cmp():
    a_np = np.random.normal(size=(5, 5))
    b_np = np.random.normal(size=(5, 5))
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
    a_np = np.random.normal(size=(5, 5))
    a_ca = ca.array(a_np)

    s_np = np.sum(a_np)
    s_ca = ca.sum(a_ca)
    print(np.allclose(s_np, np.array(s_ca)))

    a_np = np.random.normal(size=(5, 5))
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
    a_np = np.random.normal(size=(1000, 1000), loc=3, scale=10)
    a_ca = ca.random.normal(size=(1000, 1000), loc=3, scale=10)
    a_ca = np.array(a_ca)
    print(np.mean(a_np), np.mean(a_ca))
    print(np.std(a_np), np.std(a_ca))

    a_np = np.random.uniform(size=(1000, 1000), low=0, high=1)
    a_ca = ca.random.uniform(size=(1000, 1000), low=0, high=1)
    a_ca = np.array(a_ca)
    print(np.mean(a_np), np.mean(a_ca))
    print(np.std(a_np), np.std(a_ca))

    a_np = np.random.uniform(size=(1000, 1000), low=-10, high=30)
    a_ca = ca.random.uniform(size=(1000, 1000), low=-10, high=30)
    a_ca = np.array(a_ca)
    print(np.mean(a_np), np.mean(a_ca))
    print(np.std(a_np), np.std(a_ca))


def test_reduce():
    a_np = np.random.normal(size=(1024,))
    a_ca = ca.array(a_np)
    c_np = np.sum(a_np)
    c_ca = ca.sum(a_ca)
    print(np.allclose(c_np, np.array(c_ca)))
    c_np = np.mean(a_np)
    c_ca = ca.mean(a_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(5, 5))
    a_ca = ca.array(a_np)
    c_np = np.sum(a_np)
    c_ca = ca.sum(a_ca)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.sum(a_np, axis=0)
    c_ca = ca.sum(a_ca, axis=0)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.sum(a_np, axis=1)
    c_ca = ca.sum(a_ca, axis=1)
    print(np.allclose(c_np, np.array(c_ca)))

    a_np = np.random.normal(size=(5, 7, 11))
    a_ca = ca.array(a_np)
    c_np = np.sum(a_np, axis=0)
    c_ca = ca.sum(a_ca, axis=0)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.sum(a_np, axis=2)
    c_ca = ca.sum(a_ca, axis=2)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.sum(a_np, axis=(0, 1))
    c_ca = ca.sum(a_ca, axis=(0, 1))
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.sum(a_np, axis=(1, 2))
    c_ca = ca.sum(a_ca, axis=(1, 2))
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.argmin(a_np, axis=0)
    c_ca = ca.argmin(a_ca, axis=0)
    print(np.allclose(c_np, np.array(c_ca)))

    c_np = np.argmin(a_np, axis=2)
    c_ca = ca.argmin(a_ca, axis=2)
    print(np.allclose(c_np, np.array(c_ca)))


def test_indexing():
    a_np = np.ones((3, 3, 3)) * np.arange(3)
    a_ca = ca.array(a_np)

    print(np.allclose(a_np[0], np.array(a_ca[0])))
    print(np.allclose(a_np[1], np.array(a_ca[1])))
    print(np.allclose(a_np[0, :, :], np.array(a_ca[0, :, :])))
    print(np.allclose(a_np[2, :, :], np.array(a_ca[2, :, :])))
    print(np.allclose(a_np[1, 1, :], np.array(a_ca[1, 1, :])))
    print(np.allclose(a_np[1, 1, 1], np.array(a_ca[1, 1, 1])))
    print(np.allclose(a_np[1:3, :, :], np.array(a_ca[1:3, :, :])))

    a_np = np.ones((3, 3, 3)) * np.arange(3)
    a_ca = ca.array(a_np)
    b_np = np.random.random(size=(3, 3))
    b_ca = ca.array(b_np)

    a_np[1] = b_np
    a_ca[1] = b_ca
    print(np.allclose(a_np, np.array(a_ca)))

    b_np = np.random.random(size=(3))
    b_ca = ca.array(b_np)
    a_np[1, 2] = b_np
    a_ca[1, 2] = b_ca
    print(np.allclose(a_np, np.array(a_ca)))


def test_transpose():
    shapes = [(4, 4), (5, 4), (8, 8), (32, 32), (55, 44), (64, 55), (55, 64),
              (32, 64), (64, 128), (128, 64), (128, 1)]
    for shape in shapes:
        a_np = np.reshape(np.arange(np.prod(shape)), shape)+1
        a_ca = ca.array(a_np)
        a_np = np.ascontiguousarray(a_np.T)
        a_ca = ca.ascontiguousarray(a_ca.T)
        print(np.allclose(a_np, np.array(a_ca)))


def test_copyto():
    a_np = np.random.random(size=(7, 11))
    a_ca = ca.array(a_np)

    b_np = np.zeros_like(a_np)
    b_ca = np.zeros_like(a_np)
    ca.copyto(b_np, a_ca)
    print(np.allclose(a_np, b_np))
    ca.copyto(b_ca, a_np)
    print(np.allclose(np.array(a_ca), np.array(b_ca)))
    ca.copyto(b_ca, a_ca)
    print(np.allclose(np.array(a_ca), np.array(b_ca)))


def test_concatenate():
    def concatenate_(shape_a, shape_b, axis):
        a = np.random.random(size=shape_a)
        b = np.random.random(size=shape_b)
        c_np = np.concatenate((a, b), axis=axis)
        c_ca = ca.extra.concatenate(ca.array(a), ca.array(b), axis=axis)
        print(np.allclose(c_np, np.array(c_ca)))
        a_, b_ = ca.extra.split(c_ca, a_size=a.shape[axis], axis=axis)
        print(np.allclose(a, np.array(a_)))
        print(np.allclose(b, np.array(b_)))
    concatenate_((3,), (4,), axis=0)

    concatenate_((2, 3), (2, 3), axis=0)
    concatenate_((2, 3), (2, 3), axis=1)
    concatenate_((2, 3), (5, 3), axis=0)
    concatenate_((2, 3), (2, 5), axis=1)

    concatenate_((2, 3, 4), (2, 3, 4), axis=0)
    concatenate_((2, 3, 4), (2, 3, 4), axis=1)
    concatenate_((2, 3, 4), (2, 3, 4), axis=2)
    concatenate_((2, 3, 4), (5, 3, 4), axis=0)
    concatenate_((2, 3, 4), (2, 5, 4), axis=1)
    concatenate_((2, 3, 4), (2, 3, 5), axis=2)

    concatenate_((2, 3, 4, 5), (2, 3, 4, 5), axis=0)
    concatenate_((2, 3, 4, 5), (2, 3, 4, 5), axis=1)
    concatenate_((2, 3, 4, 5), (2, 3, 4, 5), axis=2)
    concatenate_((2, 3, 4, 5), (2, 3, 4, 5), axis=3)
    concatenate_((2, 3, 4, 5), (7, 3, 4, 5), axis=0)
    concatenate_((2, 3, 4, 5), (2, 7, 4, 5), axis=1)
    concatenate_((2, 3, 4, 5), (2, 3, 7, 5), axis=2)
    concatenate_((2, 3, 4, 5), (2, 3, 4, 7), axis=3)


def run():
    test_indexing()
    test_dot()
    test_multiply()
    test_binary()
    test_binary_cmp()
    test_sum()
    test_random()
    test_reduce()
    test_transpose()
    test_concatenate()


if __name__ == '__main__':
    run()
