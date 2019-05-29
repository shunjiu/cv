import Convolution
import gauss
import numpy as np 
import cv2
import math

def gausslb(img,size,sigma):
    gg = gauss.gauss(size,sigma)
    newimg = Convolution.Convolution1(img,gg)
    # cv2.imshow('gs',newimg)
    sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    gx = Convolution.Convolution1(newimg,sobelx)
    gy = Convolution.Convolution1(newimg,sobely)
    tmp_img = np.array(np.sqrt(gx**2+gy**2))
    return gx,gy,tmp_img

def gradient(gx,gy,lb_img):
    (l,w) = gx.shape
    gra = np.zeros((l,w))
    for i in range(l):
        for j in range(w):
            gra[i,j] = math.atan2(gy[i,j],gx[i,j])
    return gra
def ishigh(lb_img,i,j,highthreshold):
    if lb_img[i-1,j-1] >= highthreshold:
        return True
    elif lb_img[i-1,j] >= highthreshold:
        return True
    elif lb_img[i-1,j+1] >= highthreshold:
        return True
    elif lb_img[i,j-1] >= highthreshold:
        return True
    elif lb_img[i,j+1] >= highthreshold:
        return True
    elif lb_img[i+1,j-1] >= highthreshold:
        return True
    elif lb_img[i+1,j] >= highthreshold:
        return True
    elif lb_img[i+1,j+1] >= highthreshold:
        return True
    else:
        return False
def suppressed(lb_img,gra,highthreshold,lowthreshold):
    new_lb_img = lb_img.copy()
    (l,w) = lb_img.shape
    for i in range(1,l-1):
        for j in range(1,w-1):
            theta = abs(gra[i,j])
            if theta >=0 and theta < 45/180*math.pi:
                gp1 = (1-math.tan(theta))*lb_img[i,j+1] + math.tan(theta)*lb_img[i-1,j+1]
                gp2 = (1-math.tan(theta))*lb_img[i,j-1] + math.tan(theta)*lb_img[i+1,j-1]
            elif theta >= 45/180*math.pi and theta < 90/180*math.pi:
                gp1 = (1-math.tan(theta))*lb_img[i-1,j+1] + math.tan(theta)*lb_img[i-1,j]
                gp2 = (1-math.tan(theta))*lb_img[i+1,j-1] + math.tan(theta)*lb_img[i+1,j]
            elif theta >= 90/180*math.pi and theta < 135/180*math.pi:
                gp1 = (1-math.tan(theta))*lb_img[i-1,j] + math.tan(theta)*lb_img[i-1,j-1]
                gp2 = (1-math.tan(theta))*lb_img[i+1,j] + math.tan(theta)*lb_img[i+1,j+1]
            elif theta >= 135/180*math.pi and theta <= 180/180*math.pi:
                gp1 = (1-math.tan(theta))*lb_img[i-1,j-1] + math.tan(theta)*lb_img[i,j-1]
                gp2 = (1-math.tan(theta))*lb_img[i+1,j+1] + math.tan(theta)*lb_img[i,j+1]
            if lb_img[i,j] >= gp1 and lb_img[i,j] >= gp2:
                # new_lb_img[i,j] = lb_img[i,j] 
                if lb_img[i,j] >= highthreshold:
                    new_lb_img[i,j] = 255
                elif lb_img[i,j] >= lowthreshold:
                    if ishigh(lb_img,i,j,highthreshold) == False:
                        new_lb_img[i,j] = 0
                    else:
                        new_lb_img[i,j] = 255
                else:
                    new_lb_img[i,j] = 0
            else:
                new_lb_img[i,j] = 0

            # if lb_img[i,j] >= highthreshold:
            #     new_lb_img[i,j] = 255
            # elif lb_img[i,j] >= lowthreshold:
            #     if ishigh(lb_img,i,j,highthreshold) == False:
            #         new_lb_img[i,j] = 0
            #     else:
            #         new_lb_img[i,j] = 255
            # else:
            #     new_lb_img[i,j] = 0
    return new_lb_img
if __name__ == '__main__':
    img = cv2.imread('test2.jpg',0)
    # res,threshold = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
    gx,gy,lb_img = gausslb(img,5,1.4**2)
    gra = gradient(gx,gy,lb_img)
    process_img = suppressed(lb_img,gra,80,50)
    # np.savetxt('img.txt',lb_img)
    # print(process_img.all==lb_img.all)
    cv2.imwrite('result.jpg',process_img)

    # cv2.imshow('yt',img)
    # cv2.imwrite('res.jpg',lb_img)
    # cv2.imshow('test',lb_img)
    # cv2.imshow('',lb_img.astype('float32'))
    # cv2.waitKey(0)


