import numpy as np 
import math

def gauss(size,sigma):
    list1 = np.zeros((size,size))
    center = size//2
    sum1=0
    for i in range(size):
        tmp1 = (i-center)**2
        for j in range(size):
            tmp2 = (j-center)**2
            g=math.exp(-(tmp1+tmp2)/(2*sigma))/(2*math.pi*sigma)
            sum1+=g
            list1[i,j] = g
    return list1/sum1
if __name__ == '__main__':
    print(gauss(3,1))