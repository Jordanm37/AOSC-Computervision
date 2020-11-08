import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from spectral_2D_image_match_functions import *

    
def main():

    patternDir = "wallypuzzle_rocket_man.png"
    templateDir = "wallypuzzle_png.png"

    pattern_gray = convert_gray( np.array(read_image( patternDir ) ) )
    template_gray = convert_gray( np.array(read_image( templateDir ) ) )
    
    pattern_s = pattern_gray - np.mean(pattern_gray)
    template_s = template_gray - np.mean(template_gray)
  
    start = time.time()
    image_cross, image_cross_value = find_offset( pattern_s, template_s)
    end = time.time()

    #function to find image centre
    vertCen = pattern_gray.shape[1]/2
    horCen = pattern_gray.shape[0]/2

    #plot shift FFT of image
    # plt.subplot(2,2,1)
    plt.imshow( np.fft.fftshift( np.imag(matrix_fft(pattern_s)) ) ) 
    # plt.subplot(2,2,2)
    plt.imshow( mpimg.imread( patternDir ) )   
    # plt.subplot(2,2,3)
    plt.imshow( mpimg.imread( templateDir ) )  
    circle=plt.Circle(( image_cross[1] + vertCen ,\
    image_cross[0] + horCen  ),\
    50,facecolor='red', edgecolor='blue',linestyle='dotted', \
    linewidth='2.2')
    plt.gca().add_patch(circle)  
    plt.show()    
    plt.ion()    

    #centre of pattern
    print("Offset_x_co = ", image_cross[1] + horCen , "Offset_y_co = ", image_cross[0] + vertCen, "value =", image_cross_value)
    #top left corner of pattern image
    print("Offset_x_co = ", image_cross[1] , "Offset_y_co = ", image_cross[0] , "value =", image_cross_value)
    print("run time = ", end - start )


if __name__ == '__main__':
    
    main()

"""
Offset_x =  528 Offset_y =  982 value = 0.520092887633342
<class 'numpy.ndarray'>
"""


