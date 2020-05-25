from PIL import Image
import numpy as np
import cv2

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    window = [
        (i, j)
        for i in range(-indexer, filter_size-indexer)
        for j in range(-indexer, filter_size-indexer)
    ]
    index = len(window) // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = sorted(
                0 if (
                    min(i+a, j+b) < 0
                    or len(data) <= i+a
                    or len(data[0]) <= j+b
                ) else data[i+a][j+b]
                for a, b in window
            )[index]
    return data


def mean_filter(img):
    members = [(0,0)] * 9
    width, height = img.shape[:2]
    cv2.imshow('',img)
    cv2.waitKey(0)
    newimg = img    
    for i in range(1,width-1):
        for j in range(1,height-1):
            members[0] = img[i-1,j-1]
            members[1] = img[i-1,j]
            members[2] = img[i-1,j+1]
            members[3] = img[i,j-1]
            members[4] = img[i,j]
            members[5] = img[i,j+1]
            members[6] = img[i+1,j-1]
            members[7] = img[i+1,j]
            members[8] = img[i+1,j+1]
            total=(members[0]+members[1]+members[2]+members[3]+members[4]+members[5]+members[6]+members[7]+members[8])/9
            newimg[i,j] = total
    return newimg

def contrast_enhancement(img1):
    minmax_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            minmax_img[i,j] = 255*(img1[i,j]-np.min(img1))/(np.max(img1)-np.min(img1))
    return minmax_img

img = cv2.imread('1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("original",mat=img)
cv2.waitKey(0)

contrasted = contrast_enhancement(img)
cv2.imshow("contrasted",mat=contrasted)
cv2.waitKey(0)

medianed = median_filter(img, 3)
cv2.imshow("median",mat=medianed)
cv2.waitKey(0)

meaned = mean_filter(img)
cv2.imshow("mean",mat=meaned)
cv2.waitKey(0)


