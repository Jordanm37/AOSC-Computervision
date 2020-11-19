import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from numpy import pi, exp, sqrt
import os

def convert_gray(image):

    image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 0] + 0.1140 * image[:, :, 0]   
    
    return image

def calculate_energy( pattern, g_slice ):
    """
    Normalisation for 1D slice of N size array of same length of pattern passed. norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )
    Inputs:
        p         list    Pattern must be non empty and sum-square of its elements precalculated and passed
		g_slice   list    slice of t from i to offset
    Output: 
        norm      Scalar  Float of variance for a given slice of the template/search and pattern
     """
	
    g_slice_sq = g_slice ** 2
    g_slice_sum = g_slice_sq.sum()
    product = pattern * g_slice_sum
    norm = np.sqrt(product) 

    return norm

def calculate_score( pattern, template):
   
    score = (pattern * template).sum()
   
    return score

def zero_padding(input_array, padlen_x, padlen_y):

    m,n = input_array.shape
    padlen_x = int(np.round(padlen_x))
    padlen_y = int(np.round(padlen_y))
    padded = np.zeros((m +2*padlen_x , n+2*padlen_y ),dtype=input_array.dtype)
    padded[padlen_x:-padlen_x:, padlen_y:-padlen_y] = input_array
    
    return padded

def find_best_match( score ):

    max_element = np.amax( score )

    index = np.unravel_index(np.argmax( score, axis=None), score.shape) 

    return index, max_element 


def resize_even(pattern, template):   
    extra_row = 0
    extra_col = 0
    a = pattern
    b = template
    if a.shape[0]%2!=0:
        extra_row = 1
        a = np.vstack((a,np.zeros( (1,a.shape[1])  )))

    if a.shape[1]%2!=0:
        extra_col = 1
        a = np.hstack((a,np.zeros( (a.shape[0],1) )))

    if b.shape[0]%2!=0:
        b = np.vstack((b,np.zeros( (1,b.shape[1]) )))
        
    if b.shape[1]%2!=0:
        b = np.hstack((b,np.zeros( (b.shape[0],1) )))

    return a, b    

def n_corr2d( pattern, template):

    #Pad and initalise arrays for calculation  
    pattern, template = resize_even(pattern, template)
    template_padded = zero_padding( template, pattern.shape[0]/2, pattern.shape[1]/2 )
    side_edge = template.shape[0] - pattern.shape[0] 
    bottom_edge = template.shape[1] - pattern.shape[1]
    
    scores = np.zeros( ( side_edge ,  bottom_edge ) )
    norm =  np.zeros( ( side_edge ,  bottom_edge ) )
    norm_scores =  np.zeros( ( side_edge ,  bottom_edge ) )

    template_pad_arr = np.array( template_padded )
    pattern_arr = np.array( pattern ) 
    pattern_sq_sum = ( pattern_arr ** 2 ).sum()
    height_pattern = pattern.shape[0]
    width_pattern = pattern.shape[1]

    for i in range( scores.shape[0] ):
        for j in range( scores.shape[1] ):
            g_slice_i_j = template_pad_arr[ i : i + height_pattern, j : j + width_pattern ]
            scores_i_j = calculate_score( pattern_arr, g_slice_i_j)
            if  scores_i_j!=0 : 
                norm_i_j = calculate_energy( pattern_sq_sum, g_slice_i_j)
                norm_scores[ i, j ] = scores_i_j/norm_i_j 

    return norm_scores

def read_image(image_name):

    img = mpimg.imread(image_name)

    return img

def visualize_results(pattern,pattern_ms,template,template_ms):
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
    plt.title("Pixel intensity of pattern")
    plt.subplot(1,2,2)  
    plt.hist(lum_img_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of template")
    plt.show()

def genGausTemplate(sigma = 1.0, mu = 0.0, gridDim = 10):
    x, y = np.meshgrid(np.linspace(-1,1,gridDim), np.linspace(-1,1,gridDim))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g    

def plot_save(label):
    path = os.path.join("..","figures","calibration","fig_" + label + ".png")
    plt.savefig(path)

def dot_detection(corr_arr, img):
    maxVals = []
    threshold = 0.15
    numOfDots = 0
    dots = np.zeros( (corr_arr.shape[0], corr_arr.shape[1]) ) 
    for i in range( corr_arr.shape[0] ):
        for j in range(corr_arr.shape[1]):
                if corr_arr[i ,j] >= threshold:
                    maxVals.append(corr_arr[i ,j])  
                    dots[i,j] = 1
                    numOfDots += 1
    
    plt.imshow(corr_arr)
    plt.title("Correlation values")
    plot_save("Dots detected Convolution")
    plt.show()
    plt.hist(corr_arr.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of correlation")
    plt.show()
    plt.imshow(dots, alpha=0.3)
    plt.title("Dots")
    plt.show()
    plt.imshow(img)
    plt.imshow(dots, alpha=0.3)
    plt.title("Image with dots")
    plt.show()