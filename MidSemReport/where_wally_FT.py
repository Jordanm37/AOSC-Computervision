import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import islice
import matplotlib.image as mpimg 

    
def read_image(image_name):
    """
    Read image 
 
    Inputs:
    ----------------
        image_name   Image path 
 
    Output:
    ----------------
        img  Image as multi channel array
       """      
    img = mpimg.imread(image_name)
    #im_array = np.array(img)

    return img


#function that finds the largest element and its index in an array
def find_best_match(score):
    """
    Find max value in 2D array and its index
 
    Inputs:
    ----------------
        score   2D target array
        
    Output:
    ----------------
        index   Index of largest element 
        
        max_element Max Element in the array

     """      
    #try:
    max_element = np.amax(score)
    #except:
    #    print( "Line 45 Error", score )
    index = np.unravel_index(np.argmax( score, axis=None), score.shape) 
    #index = np.argmax(score)

    return index, max_element # tuple = list, int


def matrix_fft(pattern):
    """
    FFT of the input array
 
    Inputs:
    ----------------
        pattern   2D array
        
    Output:
    ----------------
        fft2   FFT of array

     """
    #Take FFt along columns, then rows       
    fft1 = np.fft.fft(pattern, axis = 0)
    fft2 = np.fft.fft(fft1, axis = 1)

    return fft2

def matrix_ifft(pattern):
    """
    IFFT of the input array
 
    Inputs:
    ----------------
        pattern   2D array
        
    Output:
    ----------------
        ifft2   FFT of array

     """  

    #Take IFFt along columns, then rows    
    ifft1 = np.fft.ifft(pattern, axis = 0)
    ifft2 = np.fft.ifft(ifft1, axis = 1)

    return ifft2


def matrix_complex_conj(pattern):
    """
    Complex of the input array
 
    Inputs:
    ----------------
        pattern   2D array
        
    Output:
    ----------------
        pattern_fft_conj   Complex conjugate of array

     """  

    pattern_fft_conj = np.conj(pattern)

    return pattern_fft_conj 



def zero_padding(C, x_pad, y_pad):
    """
    Zero pad 2D array by placing it in centre of zeroed matrix of padded size.
 
    Inputs:
    ----------------
        array   The array to pad
 
        padlen_x    Padwidth of the rows. Floats will be rounded up.
        
        padlen_y    Padwidth of the columns. Floats will be rounded up.
 
    Output:
    ----------------
        padded  Padded template array.  
     """        

    # m,n = c_x.shape
    
    # #needs to be int to work not float make this into a round up if float function or find libray function 
    # if padlen_x% 2 == 0:
    #     padlen_x = int(padlen_x)
    # else: 
    #     padlen_x = int( padlen_x + 0.5 )

    # if padlen_y% 2 == 0:
    #     padlen_y = int(padlen_y)
    # else: 
    #     padlen_y = int( padlen_y + 0.5 )
           
    # c_y = np.zeros((m +2*padlen_x , n+2*padlen_y ),dtype=c_x.dtype)
    # c_y[padlen_x:-padlen_x:, padlen_y:-padlen_y] = c_x
    # return c_y
    
    
    x_pad = int(np.round(x_pad))
    y_pad = int(np.round(y_pad))
    
    return np.pad(C, [(x_pad, ), (y_pad, )], mode='constant')


def crr_2d( pattern, template):
    """
    Cross correlation of two 2D arrays using FFt to convolve spatial arrays
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        real_corr  Cross correlation array
     """  

    side_edge_pad = template.shape[0] - pattern.shape[0] # move into zero padding function
    bottom_edge_pad = template.shape[1] - pattern.shape[1]

    pattern_padded = zero_padding( pattern, side_edge_pad /2, bottom_edge_pad /2 ) # pad pattern as centre of array with zeros

    template_fft = matrix_fft(template) #(a)
    pattern_fft_conj = matrix_complex_conj( matrix_fft(pattern_padded) ) # (b)

    # a * b
    #Offset pattern due to padding
    product = pattern_fft_conj[0: pattern_fft_conj.shape [0]-1,0: pattern_fft_conj.shape [1] -1] *  template      
        
    ccr = matrix_ifft(product)
    
    real_corr = np.real(ccr) #np.real

    return real_corr


def find_offset(pattern, template): 
    """
    2D array offset index and value from cross correlation 
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        (best_score, best_match)  Index of offset found from cross correlation
     """     

    real_corr = crr_2d( pattern, template)


    best_match , match_value = find_best_match( real_corr )
    #print( best_match )

    return (best_match[0] - 2 * pattern.shape[0], best_match[1] - 2 * pattern.shape[1]), match_value


def main():

    start = time.time()

    motif_image = read_image("wallypuzzle_rocket_man.png")
    test_image = read_image("wallypuzzle_png.png")

    image_mean_1= motif_image[:,:,0:3].mean(axis=2)
    image_mean_2= test_image[:,:,0:3].mean(axis=2)

    #plot shift FFT of image
    plt.imshow( np.fft.fftshift( np.imag(matrix_fft(image_mean_1)) ) ) 
    plt.show()

    image_cross, image_cross_value = find_offset( image_mean_1, image_mean_2)

    end = time.time()
    print("Offset_x = ", image_cross[0], "Offset_y = ", image_cross[1], "value =", image_cross_value)
    print("run time = ", end - start )

    plt.ion()

    #test_plot = test_image[ image_cross[0] : image_cross[0] + motif_image.shape[0],  image_cross[1] : image_cross[1] + motif_image.shape[1], : ] 
    #plt.imshow( test_plot )

if __name__ == '__main__':
    
    main()

"""
Offset_x =  528 Offset_y =  982 value = 0.520092887633342
<class 'numpy.ndarray'>
"""


