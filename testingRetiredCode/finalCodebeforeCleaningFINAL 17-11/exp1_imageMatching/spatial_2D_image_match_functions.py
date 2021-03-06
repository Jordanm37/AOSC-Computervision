import scipy as sp
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import os

def convert_gray(image):
    """
    Convert RGB image to Gray-Scale using formula:
    0.2989 * R + 0.5870 * G + 0.1140 * B
    
    Inputs:
    ----------------
        RGB Image
    
    Output:
    ----------------   
        Gray-Scale Image    
    
    """
    image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 0] + 0.1140 * image[:, :, 0]   
    
    return image

def calculate_energy( pattern, template):
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
     
    # g_slice = template[ offset_x : offset_x +pattern.shape[0],  offset_y : offset_y + pattern.shape[1]] 
    # norm = np.sqrt( ( pattern**2 ).sum() * ( g_slice**2).sum() ) 
    g_slice = template
    g_slice_squared = g_slice ** 2
    # b = a.sum()
    g_slice_sq_sum = g_slice_squared.sum()
    product = pattern * g_slice_sq_sum
    #norm = np.sqrt( p * ( g_slice**2).sum() )
    norm = np.sqrt(product) 

    #Where 0 values not caught by corr function, print to see where they occur
    # if norm == 0 :
    #     print ("p=", pattern, "template=", g_slice, "offset_x = ", offset_x, "offset_y = ", offset_y, "\n")

    return norm

def calculate_score( pattern, template):
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

    score = (pattern * template).sum()
   
    return score

def zero_padding(input_array, padlen_x, padlen_y):
    """
    Zero pad 2D input_array by placing it in centre of zeroed matrix of padded size.
 
    Inputs:
    ----------------
        input_array   The array to pad
 
        padlen_x    Padwidth of the rows. Floats will be rounded up.
        
        padlen_y    Padwidth of the columns. Floats will be rounded up.
 
    Output:
    ----------------
        padded  Padded template array.  
     """ 
     
    m,n = input_array.shape
    padlen_x = int(padlen_x)
    padlen_y = int(padlen_y)  

    padded = np.zeros((m +2*padlen_x , n+2*padlen_y ),dtype=input_array.dtype)
    padded[padlen_x:-padlen_x:, padlen_y:-padlen_y] = input_array
    
    return padded

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
     
    #try:
    max_element = np.amax( score )
    #except:
    #    print( "Line 45 Error", score )
    index = np.unravel_index(np.argmax( score, axis=None), score.shape) 

    return index, max_element 

def resize_even(pattern, template):   
    extra_row = 0
    extra_col = 0
    a = pattern
    b = template
    if a.shape[0]%2!=0:
        extra_row += 1
        a = np.vstack((a,np.zeros( (1,a.shape[1])  )))

    if a.shape[1]%2!=0:
        extra_col += 1
        a = np.hstack((a,np.zeros( (a.shape[0],1) )))

    if b.shape[0]%2!=0:
        b = np.vstack((b,np.zeros( (1,b.shape[1]) )))
        extra_row += 1

    if b.shape[1]%2!=0:
        b = np.hstack((b,np.zeros( (b.shape[0],1) )))
        extra_col += 1

    return a, b , extra_row, extra_col
       

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
    template_padded = zero_padding( template, pattern.shape[0], pattern.shape[1] )
    side_edge = template_padded.shape[0] - pattern.shape[0] 
    bottom_edge = template_padded.shape[1] - pattern.shape[1]
    
    scores = np.zeros( ( side_edge ,  bottom_edge ) )
    norm =  np.zeros( ( side_edge ,  bottom_edge ) )
    norm_scores =  np.zeros( ( side_edge ,  bottom_edge ) )
    #test = [0] * ( len( template ) - len( pattern ) )
	
    template_pad_arr = np.array( template_padded )
    pattern_arr = np.array( pattern ) 
    pattern_sq_sum = ( pattern_arr ** 2 ).sum()
    height_pattern = pattern.shape[0]
    width_pattern = pattern.shape[1]
    for i in range( scores.shape[0] ):
    #t_start = time.time()
        for j in range( scores.shape[1] ):
            g_slice_i_j = template_pad_arr[ i : i + height_pattern, j : j + width_pattern ]
            scores_i_j = calculate_score( pattern_arr, g_slice_i_j)
            #norm[ i ] = g( pattern, template, i)
            #print( scores )
            if  scores_i_j!=0 : 
                norm_i_j = calculate_energy( pattern_sq_sum, g_slice_i_j)
                norm_scores[ i, j ] = scores_i_j/norm_i_j 
        #tn = time.time()
        #print( f'{ i } run time =  { tn - t_start}')

        #print( "s=", scores,"\n", "n=", norm, "\n")

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
    pattern, template, xtr_row, xtr_col = resize_even(pattern, template)
    norm_corr = n_corr2d( pattern, template)
    best_match , match_value = find_best_match( norm_corr )
    best_x = best_match[0] - pattern.shape[0] - xtr_row + 1
    best_y = best_match[1] - pattern.shape[1] - xtr_col + 1 

    return ( best_x, best_y ), match_value
 
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

    return img

def visualize_results(pattern,pattern_ms,template,template_ms,horCen,vertCen,image_cross,):
    '''
    Visualize pattern and template images   
    '''
        
    # Intesity histogram plot
    lum_img_1 = pattern_ms[:, :]
    lum_img_2 = template_ms[:, :]  
    
    # plots of original and greyscale
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(2,2,1)
    plt.imshow(pattern)
  
    plt.subplot(2,2,2)
    plt.imshow(template)
    
    plt.subplot(2,2,3)
    plt.imshow(pattern_ms)
    
    plt.subplot(2,2,4)
    plt.imshow(template_ms)
    plot_save("wholePatternAndTemplatePlots")
    
    #plot of intensity
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.imshow(pattern_ms)
    
    plt.subplot(1,2,2)
    plt.imshow(template_ms)
    #plot_save("intensityPlot")
    
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)  
    plt.hist(lum_img_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel_intensity_of_pattern")
    plt.subplot(1,2,2)  
    plt.hist(lum_img_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of template")
    plot_save("pixelIntensityOfTemplate")
    plt.show()
    
    #plot mark where pattern is found
    plt.imshow( template )  
    circle=plt.Circle(( image_cross[1] + vertCen, image_cross[0] + horCen ), \
    50, facecolor='red', edgecolor='blue', linestyle='dotted', linewidth='2.2')
    plt.gca().add_patch(circle)  
    plot_save("template_with_match") 
    plt.show()  
    plt.ion()        
    
def plot_save(label):
    path = os.path.join("..","figures","2D_image_spatial","fig_" + label + ".png")
    plt.savefig(path)