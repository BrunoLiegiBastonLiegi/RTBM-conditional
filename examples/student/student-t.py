

# Example of a 2D multivariate student-t distribution.
# A sample of 5000 elements extracted from this distribution is contained
# in student.npy.
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
student = ut.Student(np.array([0,0]), np.array([[2,-1],[-1,4]]), 6)


# Sampling data
data = student.sampling()



# Building grid for evaluation
n = 50
xx = np.linspace(-5,5, n)
yy = np.linspace(-5,5, n)
X,Y = np.meshgrid(xx,yy)
grid = np.vstack([X,Y]).reshape(2,-1).T

# Analytic PDF
pdf = np.asarray([student.pdf(grid[i]) for i in range(n**2)]).flatten()


# RTBM model
M = rtbm.RTBM(2,2, random_bound=1, init_max_param_bound=60)

# Training
minim = minimizer.CMA(True)
#solution = minim.train(logarithmic, M, data.T, tolfun=1e-3)    # decomment if you want training to take place,
                                                                # otherwise use the pre-trained model below
                                                                
# Pre-trained model
M.set_parameters(np.array([ 1.74587313e-03, -9.13477122e-04,  8.22063600e+00,  1.74029397e+01,-1.10959475e+00,  1.02118882e+00, -6.59149043e-01,  5.99301112e-01,5.57746556e-01,  1.84310133e-01,  3.04781259e-01,  2.41516805e+01,-4.40890318e-01,  4.15665779e+01]))





# Analytic conditional
x1 = np.array([float(sys.argv[1])]).reshape(1,1)
cstudent = student.conditional(x1)
cpdf = np.asarray([cstudent.pdf(i) for i in xx]).flatten()
