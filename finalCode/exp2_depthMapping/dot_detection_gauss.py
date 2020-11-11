import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import islice
import matplotlib.image as mpimg
from numpy import pi, exp, sqrt
from FT_functions import*
from matplotlib.patches import Circle #draw on image

def gauss_2d( k ):
    probs = []
    s = 1
    k = k // 2  
    # (1/sqrt(2pi*s*s))*exp(-(z)^2/2s*s)  for u=0
    # z = -1 , 0 , 1 for k = 1
    for z in range(-k, k+1):
        probs.append(exp(-z*z/(2*s*s))/sqrt(2*pi*s*s))
    kernel = np.outer(probs, probs)

    return kernel

def main():
    gaussian = gauss_2d(3)
    img = read_image("test\\test_left_1.tiff")
    image_mean_1= img[:,:,0:3].mean(axis=2)
    corr = crr_2d(gaussian, image_mean_1)
    best = find_best_match( corr)
    max = []
    threshold = 0.001
    dots = np.zeros( (corr.shape[0], corr.shape[1] ) )   # gives a matrix of n x m with all zeros in it
    for i in range( corr.shape[0] ):
        for j in range(corr.shape[1]):
                if corr[i ,j] >= threshold:
                    max.append((i,j))  # [(1,2), (1,3), (2,4)...]
                    dots[i,j] = 1
                    # circle=plt.Circle(( i,j ),\
                    # 50,facecolor='red', edgecolor='blue',linestyle='dotted', \
                    # linewidth='2.2')

    print(dots)
    
    plt.imshow(dots)
    plt.show()
    # print(gaussian)
    # plt.imshow(gaussian)
    # plt.show()

    # print(dots)
    # print(gaussian)

    # plt.imshow(gaussian)
    # plt.title("Guassian")
    # plt.show()

    # plt.imshow(dots)
    # plt.title("Dot detected")
    # plt.show()


    # plt.imshow(img)
    # plt.gca().add_patch(circle)
    # plt.title("Overlap")
    # plt.show()


if __name__ == '__main__':
    main()
