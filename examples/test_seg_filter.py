import os
import numpy as np
import cudarray as ca



img = np.arange(8*8, dtype=np.float)

img = img.reshape((1,1,1,8,8))

img = img + 1.0

print(img)

#c_filter = np.zeros((2,2), dtype=np.float)
#c_filter[1,1] = 1.0

#c_filter = c_filter.reshape((1,1,2,2))

#c_filter = np.vstack((c_filter,c_filter))

#convLayer = ca.nsnet.ConvBC01()
#convout = convLayer.fprop(imgs=img, filters=c_filter)

#print (convout.shape)
#print (convout)

pool_layer_1 = ca.nsnet.PoolB01((2,2))

convout1 = pool_layer_1.fprop(img)

print ("\n --------------- oo -------------- \n")
print (convout1.shape)
print (convout1)


final_out = convout1
pool_layer_2 = ca.nsnet.PoolB01((2,2))

convout2 = pool_layer_2.fprop(convout1)

print (convout2)
#Flatten
print ("\n --------------- oo -------------- \n")
poolLayers = np.array((2,2))
poolLayers = poolLayers.reshape((1,2))

org = np.empty((8,8))

for i in range(poolLayers.shape[0]):
	pool_h = poolLayers[-i][0]
	pool_w = poolLayers[-i][1]






