import numpy as np
from theta import rtbm
import scipy as sp




def hist(data, a, eps=0.1, axes='x'):
    """Extracts the 1D conditional histogram.

    """
    hist = []
    if axes == 'x':
        for i in range(len(data)):
            if np.absolute(data[i,1]-a) <= eps :
                hist.append(data[i,0])                

    if axes == 'y':
        for i in range(len(data)):
            if np.absolute(data[i,0]-a) <= eps :
                hist.append(data[i,1])   

    return hist









class Student(object):
    """Multivariate student-t distribution.

    """
    def __init__(self, m, S, v, p=2):
        self.m = m
        self.S = S
        self.v = v
        self.p = p
        
    def pdf(self, x):
        pre = sp.special.gamma((self.v+self.p)/2)/(sp.special.gamma(self.v/2)*np.power(self.v*np.pi,self.p/2)*np.sqrt(np.linalg.det(self.S)))
        main = np.power(1 + (x-self.m).T.dot(sp.linalg.inv(self.S).dot(x-self.m))/self.v, -(self.v+self.p)/2)
        return pre*main

    def conditional(self, x1):
        cm = self.m[1]+self.S[1,0]*(x1-self.m[0])/self.S[0,0]
        cS = (self.v+(x1-self.m[0])**2/self.S[0,0])*(self.S[1,1]-self.S[1,0]*self.S[0,1]/self.S[0,0])/(self.v+1)
        return Student(cm, cS, self.v, p=1)

    def sampling(self, l=5, n=5000):
        data = np.zeros((n,self.p))
        ymax = sp.special.gamma((self.v+self.p)/2)/(sp.special.gamma(self.v/2)*np.power(self.v*np.pi,self.p/2)*np.sqrt(np.linalg.det(self.S)))
        count = 0
        while count < n:
            x = np.random.uniform(-l,l,self.p)
            y = np.random.uniform(0,ymax,1)
            if y < self.pdf(x):
                data[count] = x
                count = count + 1
        return data








    

def rotate(M):
    """Rotates a RTBM model by a Pi/2 angle in counterclockwise direction.
    
    """
    nh = M._Nh
    nv = M._Nv
    R = np.array([[0,-1],[1,0]])
    Rt = R.T
    Rpinv = sp.linalg.inv(Rt.dot(R)).dot(Rt)
    t = sp.linalg.inv(R.dot(sp.linalg.inv(M.t).dot(R.T)))
    w = Rpinv.T.dot(M.w).reshape(1,nh*nv)
    bv = Rpinv.T.dot(M.bv)

    rM = rtbm.RTBM(nv,nh)
    params = np.concatenate((bv, M.bh, w, t[np.triu_indices(nv)], M.q[np.triu_indices(nh)]), axis=None)
    rM.set_parameters(params)
    return rM
