#import scipy as sp
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt

def convert_gray(image):
    a = 0.2989 * image[:, :, 0]    
    b = 0.5870 * image[:, :, 0]
    c = 0.1140 * image[:, :, 0]
    image = a + b + c
    return image

def calculate_energy( pattern, template, offset_x, offset_y ):
    print(f"calculate_energy: ({offset_x}, {offset_y})")
    """
    Normalisation for 2D slice of NxM size array of same length of pattern passed.
    norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )
 
    Inputs:
    ----------------
        p          list   must be non empty and sum-square of its elements precalculated and passed

        t          list   similar dimensionality to pattern

        offset_x   int    position in the template/search array along rows

        offset_y   int    position in the template/search array along columns
        
    Output:
    ----------------
        norm       list   Scalar float of variance for a given an array slice of the template/search and pattern
     """    
    offset_x_len = offset_x +pattern.shape[0]
    offset_y_len = offset_y + pattern.shape[1]
    g_slice = template[ offset_x : offset_x_len, offset_y : offset_y_len ] 
    x = ( pattern**2 ).sum()
    y = ( g_slice**2 ).sum()
    norm = np.sqrt( x * y ) 
    #Where 0 values not caught by corr function, print to see where they occur
    if norm == 0 :
        print ("p=", pattern, "template=", g_slice, "offset_x = ", offset_x, "offset_y = ", offset_y, "\n")

    return norm

def calculate_energy_slice( pattern, g_slice, offset_x, offset_y ):
    if offset_y == 0 and offset_x % 10 == 0:
        print(f"calculate_energy_slice: ({offset_x}, {offset_y})")
    
    """
    Normalisation for 2D slice of NxM size array of same length of pattern passed.
    norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )
 
    Inputs:
    ----------------
        p          list   must be non empty and sum-square of its elements precalculated and passed

        t          list   similar dimensionality to pattern

        offset_x   int    position in the template/search array along rows

        offset_y   int    position in the template/search array along columns
        
    Output:
    ----------------
        norm       list   Scalar float of variance for a given an array slice of the template/search and pattern
     """    
    offset_x_len = offset_x +pattern.shape[0]
    offset_y_len = offset_y + pattern.shape[1]
    x = ( pattern**2 ).sum()
    y = ( g_slice**2 ).sum()
    norm = np.sqrt( x * y ) 
    #Where 0 values not caught by corr function, print to see where they occur
    if norm == 0 :
        print ("p=", pattern, "template=", g_slice, "offset_x = ", offset_x, "offset_y = ", offset_y, "\n")

    return norm

def calculate_energy_slice_pattern_precompute( pattern_square_sum, g_slice, offset_x, offset_y, patternshape0, patternshape1):
    #if offset_y == 0 and offset_x % 2 == 0:
    #print(f"calculate_energy_slice: ({offset_x}, {offset_y})")
    
    """
    Normalisation for 2D slice of NxM size array of same length of pattern passed.
    norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )
 
    Inputs:
    ----------------
        p          list   must be non empty and sum-square of its elements precalculated and passed

        t          list   similar dimensionality to pattern

        offset_x   int    position in the template/search array along rows

        offset_y   int    position in the template/search array along columns
        
    Output:
    ----------------
        norm       list   Scalar float of variance for a given an array slice of the template/search and pattern
     """    
    
    offset_x_len = offset_x +patternshape0
    offset_y_len = offset_y + patternshape1
    x = pattern_square_sum
    y = ( g_slice**2 ).sum()
    norm = np.sqrt( x * y ) 

    #Where 0 values not caught by corr function, print to see where they occur
    if norm == 0 :
        print ("p=", pattern, "template=", g_slice, "offset_x = ", offset_x, "offset_y = ", offset_y, "\n")

    return norm

def calculate_score( pattern, template, offset_x, offset_y):
    #print(f"calculate_score: ({offset_x}, {offset_y})")
    """
    Correlation for 2D slice of NxM size template/search array with pattern at given offset. Sum(f[i]*g[i+m])
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty and sum-square of its elements precalculated and passed

        template   Template with similar dimensionality to pattern

        offset  Offset position in the template/search array
        
    Output:
    ----------------
        score  Scalar float of correlation score between pattern and template slice
     """        

    offset_x_len = offset_x +pattern.shape[0]
    offset_y_len = offset_y + pattern.shape[1]
    template_slice = template[ offset_x : offset_x_len,  offset_y : offset_y_len]
    score = (pattern * template_slice).sum()
    return score

def calculate_score_slice( pattern, template_slice, offset_x, offset_y):
    #print(f"calculate_score: ({offset_x}, {offset_y})")
    """
    Correlation for 2D slice of NxM size template/search array with pattern at given offset. Sum(f[i]*g[i+m])
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty and sum-square of its elements precalculated and passed

        template   Template with similar dimensionality to pattern

        offset  Offset position in the template/search array
        
    Output:
    ----------------
        score  Scalar float of correlation score between pattern and template slice
     """        

    offset_x_len = offset_x +pattern.shape[0]
    offset_y_len = offset_y + pattern.shape[1]
    score = (pattern * template_slice).sum()
    return score

def zero_padding(array, padlen_x, padlen_y):
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
    m,n = array.shape
    padded = np.zeros((m +2*padlen_x , n+2*padlen_y ),dtype=array.dtype)
    padded[padlen_x:-padlen_x:, padlen_y:-padlen_y] = array
    return padded

#function that finds the largest element and its index in an array
def find_best_match( score ):
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
    max_element = np.amax( score )
    index = np.unravel_index(np.argmax( score, axis=None), score.shape) 
    return index, max_element 


def n_corr2d( pattern, template):
    """
    Normed cross correlation of two 2D arrays
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        norm_scores  Normed cross correlation array
     """    

    #Pad and initalise arrays for calculation  
    
    template = zero_padding( template, pattern.shape[0], pattern.shape[1] )
    side_edge = template.shape[0] - pattern.shape[0] 
    bottom_edge = template.shape[1] - pattern.shape[1]
    scores = np.zeros( ( side_edge ,  bottom_edge ) )
    norm_scores =  np.zeros( ( side_edge ,  bottom_edge ) )
    len_i = scores.shape[0]
    len_j = scores.shape[1]
    patternshape0 = pattern.shape[0]
    patternshape1 = pattern.shape[1]
    pattern_square_sum = ( pattern**2 ).sum()
    i = 0
    while i < len_i:
        if i % 10 == 0:
            print(f"i:{i}")
        j = 0
        while j < len_j:  
            g_slice = template[ i : i + pattern.shape[0], j : j + pattern.shape[1] ] 
            tmp_score = calculate_score_slice( pattern, g_slice, i, j) 
            if tmp_score != 0 : 
                tmp_norm = calculate_energy_slice_pattern_precompute( pattern_square_sum, g_slice, i, j, patternshape0, patternshape1)
                if tmp_norm != 0:
                    result = tmp_score / tmp_norm
                    norm_scores[ i, j ] = result
            j += 1
        i += 1
    return norm_scores


def find_offset(pattern, template): 
    """
    2D array offset index and value from normed cross correlation 
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        (best_score, best_match)  Index of offset found from cross correlation
     """     

    print("finding norm_corr...")
    norm_corr = n_corr2d( pattern, template)

    #Plot array of cross correlation
    # plt.figure()
    # plt.plot(norm_corr)

    #best_score, best_match = find_best_match( scores )
    print("finding best_match...")
    best_match , match_value = find_best_match( norm_corr )
    #print( best_match )

    #subtracting centred offset
    print("returning from find_offset...")
    return (best_match[0] - pattern.shape[0]  + 1, best_match[1] -  pattern.shape[1]  + 1 ), match_value

    
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