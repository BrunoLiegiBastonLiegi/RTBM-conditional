import numpy as np



def hist(data, a, eps=0.1, axes='x'):

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
