

# Example of a 2D gaussian mixture distribution.
# A sample of 5000 elements extracted from this distribution is contained
# in gaussian-mix.npy.
# We first model the distribution with a RTBM and then we can generate
# three conditional distributions at three different points, using the
# conditional function contained in tools/conditional.






import sys
sys.path.append('../../tools')

import conditional as con
import utils as ut

import numpy as np
import matplotlib.pyplot as plt
from theta import rtbm, minimizer
from theta.costfunctions import logarithmic





# loading data
n = 5000
data = np.load('gaussian-mix-sample.npy')


# RTBM model
M = rtbm.RTBM(2, 4, random_bound=1, init_max_param_bound=30)

# Training
minim = minimizer.CMA(True)
#solution = minim.train(logarithmic(), M, data.T, tolfun=1e-3)     # decomment if you want training to take place,
                                                                   # otherwise use the pre-trained model below

# Pre-trained model
M.set_parameters(np.array([ 1.02735619e-05, -3.63860337e-04,  2.15571853e+00 ,-2.96561961e-01 , 1.47561952e+01,  9.63622840e-01, -1.03625774e+01, -8.31445704e+00 , -2.55729017e-01, -7.01183003e+00 , 4.09970744e+00 ,-3.38018523e+00 , 1.70457837e+00,  6.56773389e-01 , 1.49908760e+01 , 1.15771402e+00 , 1.50000000e+01,  1.50000000e+01,  4.59256632e+00 , 3.45065143e-01 ,  2.12470448e+00 , 1.50000000e+01,  4.29228972e-02 ,-1.66464616e+00 , 1.49999786e+01,  3.06820775e-01 , 1.36879929e+01]))


# conditional models
y_slice = np.array([0.6, 0.15, -0.2]) #fixed y at which we calculate the conditional probability p(x|y)
M1 = con.conditional(M, y_slice[0])
M2 = con.conditional(M, y_slice[1])
M3 = con.conditional(M, y_slice[2])






#~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# figure for the plot

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.04

rect_scatter = [left, bottom, width, height]
rect_histx1 = [left-0.6, bottom+0.6, width*2/3, 0.2]
rect_histx2 = [left-0.6, bottom+0.25, width*2/3, 0.2]
rect_histx3 = [left-0.6, bottom-0.1, width*2/3, 0.2]

# start with a rectangular Figure
plt.figure(1, figsize=(6, 6))

axScatter = plt.axes(rect_scatter)
axHistx1 = plt.axes(rect_histx1)
axHistx2 = plt.axes(rect_histx2)
axHistx3 = plt.axes(rect_histx3)





# data 2D histogram
axScatter.hist2d(data[:,0], data[:,1], normed=True, bins=[100, 50], cmap='binary')

# grid
xx = np.linspace(axScatter.get_xlim()[0],axScatter.get_xlim()[1], 120)
yy = np.linspace(axScatter.get_ylim()[0],axScatter.get_ylim()[1], 120)
X,Y = np.meshgrid(xx,yy)
grid = np.vstack([X,Y]).reshape(2,-1)

# trained model
axScatter.contour(X, Y, M(grid).reshape(X.shape[0], Y.shape[0]),colors='blue', linestyles='--')
    
# conditional plots
axHistx1.plot(xx, M1(xx.reshape(1,120)).flatten(), color='b', linestyle='-')
axHistx2.plot(xx,M2(xx.reshape(1,120)).flatten(), color='b', linestyle='-')
axHistx3.plot(xx,M3(xx.reshape(1,120)).flatten(), color='b', linestyle='-')

# conditional histograms ( i.e. empirical conditional)
axHistx1.hist(ut.hist(data, y_slice[0]), bins='auto', density=True, color='lightgray')
axHistx2.hist(ut.hist(data, y_slice[1]), bins='auto', density=True, color='lightgray')
axHistx3.hist(ut.hist(data, y_slice[2]), bins='auto', density=True, color='lightgray')

plt.savefig('gaussian-mix.png', bbox_inches='tight')
print('# Image saved at ./gaussian-mix.png')
