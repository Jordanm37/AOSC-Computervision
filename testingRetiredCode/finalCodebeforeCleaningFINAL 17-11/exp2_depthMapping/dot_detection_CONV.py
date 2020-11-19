import numpy as np
import matplotlib.pyplot as plt
import time
import os
from dot_detection_conv_functions import *

def main():
    gaussian = genGausTemplate(1,0,40)
    img = read_image("test_images\\test_left_1.tiff")
    pattern = gaussian
    template = img

    plt.imshow( gaussian )
    plt.show()

    pattern_gray = pattern
    template_gray = convert_gray(template) 

    # mean shift
    pattern_ms = pattern_gray - np.mean(pattern_gray)
    template_ms = template_gray - np.mean(template_gray)
    visualize_results(pattern, pattern_ms, template, template_ms)


    #Dot detection
    corr = n_corr2d(pattern_ms, template_ms)
    dot_detection(corr, img)

if __name__ == '__main__':
    main()
