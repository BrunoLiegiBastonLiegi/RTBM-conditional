

# Example of a 2D multivariate student-t distribution.
# We first model the distribution with a RTBM and then we can generate
# three conditional distributions at three different points, using the
# conditional function contained in tools/conditional.
# For the student distribution is avaliable the conditional probability
# in analytic form, for more informations please refer to https://arxiv.org/abs/1604.00561





import sys
sys.path.append('../../tools')

import conditional as con
import utils as ut

import numpy as np
import matplotlib.pyplot as plt
from theta import rtbm, minimizer
from theta.costfunctions import logarithmic






# Defining the distribution
m = np.array([0,0])
S = np.array([[2,-1],[-1,4]])
v = 6
student = ut.Student(m, S, v)

# Sampling data
data = student.sampling()


# Building grid for evaluation
n = 50
xx = np.linspace(-5,5, n)
yy = np.linspace(-5,5, n)
X,Y = np.meshgrid(xx,yy)
grid = np.vstack([X,Y]).reshape(2,-1)

# Analytic pdf
pdf = np.asarray([student.pdf(grid.T[i]) for i in range(n**2)]).flatten()


# RTBM model
M = rtbm.RTBM(2,2, random_bound=1, init_max_param_bound=60)

# Training
minim = minimizer.CMA(True)
#solution = minim.train(logarithmic, M, data.T, tolfun=1e-3)    # decomment if you want training to take place,
                                                                # otherwise use the pre-trained model below
                                                                
# Pre-trained model
M.set_parameters(np.array([ 1.74587313e-03, -9.13477122e-04,  8.22063600e+00,  1.74029397e+01,-1.10959475e+00,  1.02118882e+00, -6.59149043e-01,  5.99301112e-01,5.57746556e-01,  1.84310133e-01,  3.04781259e-01,  2.41516805e+01,-4.40890318e-01,  4.15665779e+01]))



# Analytic conditional
x = np.array([-1, 0.1, 1.9]) # fixed x at which we calculate the conditional probability p(y|x)

cstudent1 = student.conditional(x[0].reshape(1,1))
cstudent2 = student.conditional(x[1].reshape(1,1))
cstudent3 = student.conditional(x[2].reshape(1,1))

cpdf1 = np.asarray([cstudent1.pdf(i) for i in xx]).flatten()
cpdf2 = np.asarray([cstudent2.pdf(i) for i in xx]).flatten()
cpdf3 = np.asarray([cstudent3.pdf(i) for i in xx]).flatten()


# Conditional models
# To generate the conditionals for fixed x we first need to rotate the RTBM
# model by a Pi/2 angle

rM = ut.rotate(M)

cM1 = con.conditional(rM, -x[0]) 
cM2 = con.conditional(rM, -x[1])
cM3 = con.conditional(rM, -x[2])








#~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Figure for the plot

# Definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.04

rect_scatter = [left, bottom, width, height]
rect_histx1 = [left-0.6, bottom+0.6, width*2/3, 0.2]
rect_histx2 = [left-0.6, bottom+0.25, width*2/3, 0.2]
rect_histx3 = [left-0.6, bottom-0.1, width*2/3, 0.2]

# Start with a rectangular Figure
plt.figure(1, figsize=(6, 6))

axScatter = plt.axes(rect_scatter)
axHistx1 = plt.axes(rect_histx1)
axHistx2 = plt.axes(rect_histx2)
axHistx3 = plt.axes(rect_histx3)





# Data 2D histogram
axScatter.hist2d(data[:,0], data[:,1], normed=True, bins=[100, 50], cmap='binary')

# Analytic pdf
axScatter.contour(X,Y,pdf.reshape(X.shape[0], Y.shape[0]), colors='red', linestyles='--')

# Trained model
axScatter.contour(X, Y, M(grid).reshape(X.shape[0], Y.shape[0]), colors='blue', linestyles='--')


# Conditional plots
# Analytic
axHistx1.plot(xx, cpdf1, color='r', linestyle='--')
axHistx2.plot(xx, cpdf2, color='r', linestyle='--')
axHistx3.plot(xx, cpdf3, color='r', linestyle='--')

# Models
axHistx1.plot(xx, cM1(xx.reshape(1,n)).flatten(), color='b', linestyle='--')
axHistx2.plot(xx, cM2(xx.reshape(1,n)).flatten(), color='b', linestyle='--')
axHistx3.plot(xx, cM3(xx.reshape(1,n)).flatten(), color='b', linestyle='--')

# Conditional histograms ( i.e. empirical conditional)
axHistx1.hist(ut.hist(data, x[0], axes='y'), bins='auto', density=True, color='lightgray')
axHistx2.hist(ut.hist(data, x[1], axes='y'), bins='auto', density=True, color='lightgray')
axHistx3.hist(ut.hist(data, x[2], axes='y'), bins='auto', density=True, color='lightgray')



plt.savefig('student-t.png', bbox_inches='tight')
print('# Image saved at ./student-t.png')
