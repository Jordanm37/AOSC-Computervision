import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from numpy import pi, exp, sqrt


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
    # b = a.sum()
    g_slice_sum = g_slice_sq.sum()
    product = pattern * g_slice_sum
    #norm = np.sqrt( p * ( g_slice**2).sum() )
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
    #test = [0] * ( len( template ) - len( pattern ) )
	
    template_pad_arr = np.array( template_padded )
    pattern_arr = np.array( pattern ) 
    pattern_sq_sum = ( pattern_arr ** 2 ).sum()
    height_pattern = pattern.shape[0]
    width_pattern = pattern.shape[1]
    #make look like 1D
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

def read_image(image_name):

    img = mpimg.imread(image_name)
    #im_array = np.array(img)

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


def gauss_2d( k ):
    probs = []
    s = 1 #standard deviation
    k = k // 2   # 17 / 9 = 1.99   17 // 9 = 1       int(17/9)    3//2 = 1
    # (1/sqrt(2pi*s*s))*exp(-(z)^2/2s*s)  for u=0
    # z = -1 , 0 , 1 for k = 1
    for z in range(-k, k+1):
        probs.append(exp(-z*z/(2*s*s))/sqrt(2*pi*s*s))
    kernel = np.outer(probs, probs)

    return kernel

def plot_save(label):
    path = os.path.join("..","figures","calibration","fig_" + label + ".png")
    plt.savefig(path)

def main():
    gaussian = gauss_2d(40)
    img = read_image("test_images\\test_left_1.tiff")
    print(img.shape)
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

    time_s = time.time()
    corr = n_corr2d(pattern_ms, template_ms)
    time_e = time.time() - time_s
    print(f'time = {time_e}')
    print( find_best_match((corr)) )
    plt.imshow(corr)
    plt.title("Correlation values")
    plot_save("Dots detected Convolution")
    plt.show()
    plt.hist(corr.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of correlation")
    plt.show()



    maxVals = []
    threshold = 0.15
    numOfDots = 0
    dots = np.zeros( (corr.shape[0], corr.shape[1]) )  # gives a matrix of n x m with all zeros in it
    for i in range( corr.shape[0] ):
        for j in range(corr.shape[1]):
                if corr[i ,j] >= threshold:
                    maxVals.append(corr[i ,j])  # [(1,2), (1,3), (2,4)...]
                    dots[i,j] = 1
                    numOfDots += 1
    
    print(dots)
    print(gaussian)
    print(numOfDots)
    print(max(maxVals))

    #plt.imshow(gaussian)

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
