import gauss
import cv2
import numpy as np

def Convolution1(img,model):
    length = img.shape[0]
    width = img.shape[1]
    (gsize1,gsize2) = model.shape
    gsize1 = gsize1//2
    gsize2 = gsize2//2
    model = np.rot90(model,2)
    newimg = []
    for i in range(gsize1,length-gsize1):
        tmp = []
        for j in range(gsize2,width-gsize2):
            tmp.append((img[i-gsize1:i+gsize1+1,j-gsize2:j+gsize2+1]*model).sum())
        newimg.append(tmp)
    newimg = np.array(newimg)
    return newimg

def Convolution3(img,model):
    length = img.shape[0]
    width = img.shape[1]
    newimg = img.copy()
    (gsize1,gsize2) = model.shape
    gsize1 = gsize1//2
    gsize2 = gsize2//2
    model = np.rot90(model,2)
    for i in range(gsize1,length-gsize1):
        for j in range(gsize2,width-gsize2):
            newimg[i,j,0] = (img[i-gsize1:i+gsize1+1,j-gsize2:j+gsize2+1,0]*model).sum()
            newimg[i,j,1] = (img[i-gsize1:i+gsize1+1,j-gsize2:j+gsize2+1,1]*model).sum()
            newimg[i,j,2] = (img[i-gsize1:i+gsize1+1,j-gsize2:j+gsize2+1,2]*model).sum()
    return newimg
if __name__ == '__main__':
    img = cv2.imread('test2.jpg')
    gg = gauss.gauss(5,0.1)
    newimg = Convolution3(img,gg)
    cv2.imwrite('test.jpg',newimg)
    # cv2.waitKey(0)