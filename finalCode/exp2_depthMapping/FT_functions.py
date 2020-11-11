import matplotlib.image as mpimg 
import numpy as np
import time
import matplotlib.pyplot as plt

def convert_gray(image):

    image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 0] + 0.1140 * image[:, :, 0]
    
    return image

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

def zero_padding(input_array, x_pad, y_pad):
    """
    Zero pad 2D array by placing it in centre of zeroed matrix of padded size.
 
    Inputs:
    ----------------
        input_array   The array to pad
 
        x_pad    Padwidth of the rows. Floats will be rounded up.
        
        y_pad    Padwidth of the columns. Floats will be rounded up.
 
    Output:
    ----------------
        padded  Padded template array.  
     """        

    m,n = input_array.shape
    
    #needs to be int to work not float make this into a round up if float function or find libray function 
    if x_pad% 2 == 0:
        x_pad = int(x_pad)
    else: 
        x_pad = int( x_pad + 0.5 )

    if y_pad% 2 == 0:
        y_pad = int(y_pad)
    else: 
        y_pad = int( y_pad + 0.5 )
           
    c_y = np.zeros((m +2*x_pad , n+2*y_pad ),dtype=input_array.dtype)
    c_y[x_pad:-x_pad:, y_pad:-y_pad] = input_array
    return c_y
       
    # x_pad = int(np.round(x_pad))
    # y_pad = int(np.round(y_pad))
    
    # return np.pad(input_array, [(x_pad, ), (y_pad, )], mode='constant')

def nextpow2(number):

    """get the next power of 2 that's greater than n"""
    number_log = np.log2(number)
    number_log = np.ceil(number_log)

    return 2 ** number_log

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
    '''Old padding'''
    
    # move into zero padding function
    side_edge_pad = template.shape[0] - pattern.shape[0] 
    bottom_edge_pad = template.shape[1] - pattern.shape[1]

    # pad pattern as centre of array with zeros
    pattern_padded = zero_padding( pattern, side_edge_pad / 2, bottom_edge_pad / 2) 

    template_fft = matrix_fft(template) #(a)
    pattern_fft_conj = matrix_complex_conj( matrix_fft(pattern_padded) ) # (b)

    #a * b. Offset pattern due to padding
    width = pattern_fft_conj.shape[0]
    height = pattern_fft_conj.shape[1]
    product = pattern_fft_conj[0:width, 0:height-1] *  template[0:width, 0:height]      
        
    ccr = matrix_ifft(product)    
    real_corr = np.real(ccr) 

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

    '''
    new resizing for odd sides
    '''
    
    extra_row = 0
    extra_col = 0
    
    tempPattern = pattern
    tempTemplate = template
    
    if tempPattern.shape[0] % 2 != 0:
        extra_row = 1
        tempPattern = np.vstack((tempPattern,np.zeros((1, tempPattern.shape[1]))))

    if tempPattern.shape[1] % 2 != 0:
        extra_col = 1
        tempPattern = np.hstack((tempPattern,np.zeros((tempPattern.shape[0],1))))

    if tempTemplate.shape[0] % 2 != 0:
        tempTemplate = np.vstack((tempTemplate,np.zeros((1,tempTemplate.shape[1]))))
        
    if tempTemplate.shape[1]%2!=0:
        tempTemplate = np.hstack((tempTemplate,np.zeros((tempTemplate.shape[0],1))))

    pattern = tempPattern
    template = tempTemplate

    real_corr = crr_2d(pattern, template) 

    best_match, match_value = find_best_match(real_corr)
    
    firstVal = best_match[0] - 2 * pattern.shape[0] - extra_row
    secondVal = best_match[1] - 2 * pattern.shape[1] - extra_col
    
    return (firstVal, secondVal), match_value