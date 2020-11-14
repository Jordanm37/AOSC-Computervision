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
    gaussian = gauss_2d(40)
    img = read_image("test_images\\test_left_1.tiff")

    pattern = gaussian
    template = img

    pattern_gray = pattern
    template_gray = convert_gray(template) 

    # mean shift
    pattern_ms = pattern_gray - np.mean(pattern_gray)
    template_ms = template_gray - np.mean(template_gray)
    print(template_ms)
    visualize_results(pattern, pattern_ms, template, template_ms)

    print(f'Pattern shape: {pattern_ms.shape}')

    # corr = crr_2d(pattern_ms, template_ms)
    corr = crr_2d(pattern_gray, template_gray)
    print( find_best_match((corr)) )
    plt.imshow(corr)
    plt.title("Correlation values")
    plt.show()
    plt.hist(corr.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of correlation")
    plt.show()

    maxVals = []
    threshold = 0.007
    numOfDots = 0
    dots = np.zeros( (corr.shape[0], corr.shape[1]) )  # gives a matrix of n x m with all zeros in it
    for i in range( corr.shape[0] ):
        for j in range(corr.shape[1]):
                if corr[i ,j] >= threshold:
                    maxVals.append(corr[i ,j])  # [(1,2), (1,3), (2,4)...]
                    dots[i,j] = 1
                    numOfDots += 1
    
    print("Dots; Gaussian; numdots; maxvalue; template shape")
    print(dots)
    print(gaussian)
    print(numOfDots)
    print(max(maxVals))
    print(img.shape)
      
    
    plt.imshow(gaussian)
    plt.title("Gaussian")
    plt.show()
    plt.imshow(img)
    plt.title("Image")
    plt.show()
    plt.imshow(dots, alpha=0.3)
    plt.title("Dots")
    plt.show()
    plt.imshow(img)
    plt.imshow(dots, alpha=0.3)
    plt.title("Image with dots")
    plt.show()

if __name__ == '__main__':
    main()
