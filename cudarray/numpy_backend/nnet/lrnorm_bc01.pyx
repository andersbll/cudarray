from __future__ import division
import numpy as np
import cython
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef Py_ssize_t uint

@cython.boundscheck(False)
@cython.wraparound(False)
def lrnorm__bc01(np.ndarray[DTYPE_t, ndim=4] imgs,
              uint N,
              DTYPE alpha,
              DTYPE beta,
              np.ndarray[DTYPE_t, ndim=4] normout = None):
	"""
	imgs has shape (n_imgs, n_channels, img_h, img_w)
	"""

	cdef DTYPE nSum 
	cdef uint n_imgs = imgs.shape[0]
	cdef uint n_channels = filters.shape[1]
    cdef uint img_h = imgs.shape[2]
    cdef uint img_w = imgs.shape[3]

    cdef uint n_up = N // 2
    cdef uint n_down = N // 2

    cdef uint max_channel
    cdef uint min_channel

    for i in range(n_imgs):
    	for y in range(img_h):
    		for x in range(img_w):
    			nSum = 0
    			for a in range(n_up):
    				nSum += imgs[i, a, y, x]

    			for c in range(n_channels):
    				#Normalazation 
    				normout[i, c, y, x] = imgs[i, c, y, x] / (((1 + alpha) * nSum) ** beta)
    				#Move the window for next channel
    				max_channel = n_up + c + 1
    				min_channel = c - n_down + 1
    				#Move the sum if more channels exists
    				if (max_channel < n_channels):
    					nSum += imgs[i, max_channel, y, x]
    				#Remove privius channel from sum
    				if (min_channel > 0):
    					nSum -= imgs[i, min_channel-1, y, x]

    return normout

