from scipy.fftpack import dct as DCT, idct
from math import cos, pi
import cv2
import time
import numpy as np

IMAGE_PATH = './pictures/bg-60.jpg'

img = cv2.imread(IMAGE_PATH)

IMAGE_HEIGHT, IMAGE_WIDTH, C = img.shape

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# gets the Y component from ycbcr
def get_luminance(img):
    return img[:, :, [0]]

# gets the cb and cr component from ycbcr
def get_chrominance(img):
    return img[:, :, [1,2]]

def summation(i, n, op):
    for i in range(n-i):
        op += op
    return op

def rgb_to_ycbcr(img):
    ycbcr = img.dot(
                np.array([
                    [ 0.299,  -0.1687,  0.5],
                    [ 0.587,  -0.3313, -0.4187],
                    [ 0.114,   0.5,    -0.0813]
                 ])
            )

    ycbcr[:,:,[1,2]] += 128

    return np.uint8(ycbcr)

# removing the higher frequency values from the dct-encoded image
def quantise_img(img):
    pass

def C(i, j):
    pass

def dct_matrix(img):
    pass

def C(i):
    return 1/pow(2,.5) if i == 0 else 1

def dct_calc(u, v):
    M = IMAGE_HEIGHT
    N = IMAGE_WIDTH
    result = 0

    for i in range(0, M):
        for j in range(0, N):
            result += C(i) * C(j) * cos(pi*u*(2*i+1)/2*N) * cos(pi*v*(2*j+1)/2*M) * img[i][j][0]

    return pow(2/N, .5) * pow(2/M, .5) * result

# discreate-cosine transform
def dct(img_luminance):
    # initialising the dct_coef nparray
    dct_coef = np.array([[0 for i in range(len(img_luminance[j]))] for j in range(len(img_luminance))])

    for u in range(0, len(img_luminance)-1):
        for v in range(0, len(img_luminance[u])-1):
            dct_coef[u][v] = dct_calc(u,v)

    return dct_coef

#divide the image into n/n blocks
def image_to_blocks(img, N):
    for i in range(img.size):
        for j in range(img[i].size):
            pass

img = rgb_to_ycbcr(img)

img_luminance = get_luminance(img)
img_chrominance = get_chrominance(img)

def dct2(a):
    return DCT(DCT(a.T, norm='ortho').T, norm='ortho')

# print(dct(img_luminance))
# luminance = dct_encoding(luminance)
print(dct2(img_luminance))

