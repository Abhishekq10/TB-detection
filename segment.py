'''
Segmenting image, converted to greys
'''
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
#matplotlib inline
from scipy import ndimage
from sklearn.cluster import KMeans

IMAGE = cv2.imread('1.png')
print("Size of image is: ", IMAGE.shape)
cv2.imshow("Image", IMAGE)
GRAY = rgb2gray(IMAGE)
cv2.imshow("gray", GRAY)
cv2.waitKey(0)
print("Size of image: ", GRAY.shape)

GRAY_RESHAPED = GRAY.reshape(GRAY.shape[0]*GRAY.shape[1])
for i in range(GRAY_RESHAPED.shape[0]):
    if GRAY_RESHAPED[i] > GRAY_RESHAPED.mean():
        GRAY_RESHAPED[i] = 1
    else:
        GRAY_RESHAPED[i] = 0
GRAY = GRAY_RESHAPED.reshape(GRAY.shape[0], GRAY.shape[1])
cv2.imshow("gray", GRAY)
cv2.waitKey(0)

GRAY = rgb2gray(IMAGE)
GRAY_RESHAPED = GRAY.reshape(GRAY.shape[0]*GRAY.shape[1])
for i in range(GRAY_RESHAPED.shape[0]):
    if GRAY_RESHAPED[i] > GRAY_RESHAPED.mean():
        GRAY_RESHAPED[i] = 3
    elif GRAY_RESHAPED[i] > 0.5:
        GRAY_RESHAPED[i] = 2
    elif GRAY_RESHAPED[i] > 0.25:
        GRAY_RESHAPED[i] = 1
    else:
        GRAY_RESHAPED[i] = 0
GRAY = GRAY_RESHAPED.reshape(GRAY.shape[0], GRAY.shape[1])
cv2.imshow("gray", GRAY)
cv2.waitKey(0)

'''
Edge detection segmentation
'''
GRAY = rgb2gray(IMAGE)

# defining the sobel filters
SOBEL_HORIZONTAL = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(SOBEL_HORIZONTAL, 'is a kernel for detecting horizontal edges')

SOBEL_VERTICAL = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(SOBEL_VERTICAL, 'is a kernel for detecting vertical edges')

OUT_H = ndimage.convolve(GRAY, SOBEL_HORIZONTAL, mode='reflect')
OUT_V = ndimage.convolve(GRAY, SOBEL_VERTICAL, mode='reflect')
# here mode determines how the input array is extended when the filter overlaps a border.


cv2.imshow('vertical', OUT_V)
cv2.waitKey(0)
cv2.imshow('horizontal', OUT_H)
cv2.waitKey(0)

KERNEL_LAPLACE = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
print(KERNEL_LAPLACE, 'is a laplacian kernel')
OUT_LAPLACE = ndimage.convolve(GRAY, KERNEL_LAPLACE, mode='reflect')
plt.imshow(OUT_LAPLACE, cmap='gray')
cv2.waitKey(0)

'''
Clustering segmentation
'''

PIC = cv2.imread('1.png')/255  # dividing by 255 to bring the pixel values between 0 and 1
print(PIC.shape)
plt.imshow(PIC)
PIC_N = PIC.reshape(PIC.shape[0]*PIC.shape[1], PIC.shape[2])
print(PIC_N.shape)

K_MEANS = KMeans(n_clusters=5, random_state=0).fit(PIC_N)
PICS_TO_SHOW = K_MEANS.cluster_centers_[K_MEANS.labels_]
CLUSTER_PIC = PICS_TO_SHOW.reshape(PIC.shape[0], PIC.shape[1], PIC.shape[2])
cv2.imshow("cluster_pic", CLUSTER_PIC)
cv2.waitKey(0)
