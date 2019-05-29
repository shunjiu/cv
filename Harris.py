import Convolution as cl
import gauss
import numpy as np 
import cv2
import math

def gausslb(img,size,sigma):
    gg = gauss.gauss(size,sigma)
    newimg = cl.Convolution1(img,gg)
    # cv2.imshow('gs',newimg)
    sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    gx = cl.Convolution1(newimg,sobelx)
    gy = cl.Convolution1(newimg,sobely)
    gx2 = cl.Convolution1(gx**2,gg)
    gxy = cl.Convolution1(gx*gy,gg)
    gy2 = cl.Convolution1(gy**2,gg)
    # print(gx.max())
    # print(gy.max())
    # cv2.imshow('gy2',gy2)
    # cv2.imshow('gxy',gxy)
    # cv2.imshow('gx2',gx2)
    return gx,gy,gx2,gxy,gy2

def getc(gx2,gxy,gy2):
    k = 0.05
    c = gx2*gy2-gxy**2 - k*(gx2+gy2)**2
    # cv2.imshow('c',c)
    return c

def ismax(l,i,j):
    if l[i,j] < l[i-1,j] or l[i,j] < l[i-1,j-1] or l[i,j] < l[i-1,j+1] or l[i,j] < l[i,j-1] or l[i,j] < l[i,j+1] or l[i,j] <l[i+1,j-1] or l[i,j] <l[i+1,j] or l[i,j] <l[i+1,j+1]:
        return False
    else:
        return True

def suppressed(img,lb_img,gx,gy,k):
    # print(lb_img.max())
    # (l,w) = gx.shape
    # gra = np.zeros((l,w))
    # for i in range(l):
    #     for j in range(w):
    #         gra[i,j] = math.atan2(gy[i,j],gx[i,j])
    max_lib = lb_img.max()
    new_lb_img = img.copy()
    (l,w) = lb_img.shape
    for i in range(1,l-1):
        for j in range(1,w-1):
            # theta = abs(gra[i,j])
            # if theta >=0 and theta < 45/180*math.pi:
            #     gp1 = (1-math.tan(theta))*lb_img[i,j+1] + math.tan(theta)*lb_img[i-1,j+1]
            #     gp2 = (1-math.tan(theta))*lb_img[i,j-1] + math.tan(theta)*lb_img[i+1,j-1]
            # elif theta >= 45/180*math.pi and theta < 90/180*math.pi:
            #     gp1 = (1-math.tan(theta))*lb_img[i-1,j+1] + math.tan(theta)*lb_img[i-1,j]
            #     gp2 = (1-math.tan(theta))*lb_img[i+1,j-1] + math.tan(theta)*lb_img[i+1,j]
            # elif theta >= 90/180*math.pi and theta < 135/180*math.pi:
            #     gp1 = (1-math.tan(theta))*lb_img[i-1,j] + math.tan(theta)*lb_img[i-1,j-1]
            #     gp2 = (1-math.tan(theta))*lb_img[i+1,j] + math.tan(theta)*lb_img[i+1,j+1]
            # elif theta >= 135/180*math.pi and theta <= 180/180*math.pi:
            #     gp1 = (1-math.tan(theta))*lb_img[i-1,j-1] + math.tan(theta)*lb_img[i,j-1]
            #     gp2 = (1-math.tan(theta))*lb_img[i+1,j+1] + math.tan(theta)*lb_img[i,j+1]
            # if lb_img[i,j] >= gp1 and lb_img[i,j] >= gp2:
            if ismax(lb_img,i,j) == True:
                # new_lb_img[i,j] = lb_img[i,j] 
                if lb_img[i,j] >= k*max_lib:
                    # new_lb_img[i,j] = 0
                    cv2.circle(new_lb_img,(j+6,i+6),3,0,1)

                    # pass
                # elif lb_img[i,j] >= lowthreshold:
                #     if ishigh(lb_img,i,j,highthreshold) == False:
                #         new_lb_img[i,j] = 0
                #     else:
                #         new_lb_img[i,j] = 255
            #     else:
            #         new_lb_img[i,j] = img[i,j]
            # else:
            #     new_lb_img[i,j] = img[i,j]


    return new_lb_img

if __name__ == '__main__':
    img = cv2.imread('1.png',0)
    # cv2.imshow('1',img)
    gx,gy,gx2,gxy,gy2 = gausslb(img,5,1.4**2)
    c = getc(gx2,gxy,gy2)
    process_img = suppressed(img,c,gx,gy,0.01)
    # cv2.imwrite('res.jpg',lb_img)
    cv2.imwrite('result.jpg',process_img)
    # cv2.waitKey(0)
