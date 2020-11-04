import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from numpy import pi, exp, sqrt



def calculate_energy( pattern, template, offset_x, offset_y ):
    """
    Normalisation for 2D slice of NxM size array of same length of pattern passed.
    norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )

    Inputs:
    ----------------
        p   Pattern must be non empty and sum-square of its elements precalculated and passed

        t   Template with similar dimensionality to pattern

        offset_x  Offset position in the template/search array along rows

        offset_y  Offset position in the template/search array along columns

    Output:
    ----------------
        norm  Scalar float of variance for a given an array slice of the template/search and pattern
     """
    g_slice = template[ offset_x : offset_x +pattern.shape[0],  offset_y : offset_y + pattern.shape[1]]
    norm = np.sqrt( ( pattern**2 ).sum() * ( g_slice**2).sum() )
    #Where 0 values not caught by corr function, print to see where they occur
    if norm == 0 :
        print ("p=", pattern, "template=", g_slice, "offset_x = ", offset_x, "offset_y = ", offset_y, "\n")

    return norm



def calculate_score( pattern, template, offset_x, offset_y):
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

    score = (pattern * template[ offset_x : offset_x +pattern.shape[0],  offset_y : offset_y + pattern.shape[1]] ).sum()

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
    #try:
    max_element = np.amax( score )
    #except:
    #    print( "Line 45 Error", score )
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
    #
    template = zero_padding( template, pattern.shape[0], pattern.shape[1] )
    side_edge = template.shape[0] - pattern.shape[0]
    bottom_edge = template.shape[1] - pattern.shape[1]

    scores = np.zeros( ( side_edge ,  bottom_edge ) )
    norm =  np.zeros( ( side_edge ,  bottom_edge ) )
    norm_scores =  np.zeros( ( side_edge ,  bottom_edge ) )
    #test = [0] * ( len( template ) - len( pattern ) )



    for i in range( scores.shape[0] ):
        #t_start = time.time()
        for j in range( scores.shape[1] ):
            scores[ i, j ] = calculate_score( pattern, template, i, j)
            #norm[ i ] = g( pattern, template, i)
            #print( scores )
            if  scores[i,j]!=0 :
                norm[ i, j ] = calculate_energy( pattern, template, i, j)
                norm_scores[ i, j ] = scores[ i, j ]/norm[ i , j ]
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

    norm_corr = n_corr2d(pattern, template)

    #Plot array of cross correlation
    plt.figure()
    plt.plot(norm_corr)

    #best_score, best_match = find_best_match( scores )
    best_match, match_value = find_best_match(norm_corr)
    #print( best_match )

    #subtracting centred offset
    return (best_match[0] - pattern.shape[0] + 1, best_match[1] - pattern.shape[1] + 1), match_value


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


def gauss_2d(k):
    probs = []
    s = 1
    k = k // 2   # 17 / 9 = 1.99   17 // 9 = 1       int(17/9)    3//2 = 1
    # (1/sqrt(2pi*s*s))*exp(-(z)^2/2s*s)  for u=0
    # z = -1 , 0 , 1 for k = 1
    for z in range(-k, k+1):
        probs.append(exp(-z*z/(2*s*s))/sqrt(2*pi*s*s))
    kernel = np.outer(probs, probs)

    return kernel


gaussian = gauss_2d(3)
img = read_image("test_left_1.tiff")
image_mean_1= img[:,:,0:3].mean(axis=2)
corr = n_corr2d(gaussian, image_mean_1)


max = []
threshold = 0.001
# gives a matrix of n x m with all zeros in it
dots = np.zeros((corr.shape[0], corr.shape[1]))
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        if corr[i, j] >= threshold:
            max.append((i, j))  # [(1,2), (1,3), (2,4)...]
            dots[i, j] = 1

print(dots)
print(gaussian)

plt.imshow(gaussian)
plt.title("Guassian")
plt.show()

plt.imshow(dots)
plt.title("Dot detected")
plt.show()


plt.imshow(img)
plt.imshow(dots, alpha=0.3)
plt.title("Overlap")


plt.show()

################################################
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

    # a * b = FFT_conj(pattern) * FFT(temaplte)
    #Offset pattern due to padding
    product = pattern_fft_conj[0: pattern_fft_conj.shape [0],0: pattern_fft_conj.shape [1]] *  template[:, 0: template.shape[1] -1 ]

    #covolution = IFFT ( FFT_conj(pattern) * FFT(template) )
    ccr = matrix_ifft(product)

    real_corr = np.real(ccr) #np.real

    return real_corr



print(dots)
print(gaussian)

#plt.imshow(gaussian)

plt.imshow(img)
plt.imshow(dots, alpha=0.3)


plt.show()


read the image
figure out z axis
create a 2d array 17, 21


from left image (x, y) location of dot -- input 1
from right image (x, y) location of dot -- input 2

output real worl coord of point (x, y, z) -- output


def generate_real_world_coords(depth, width, height):
    coords = []
    for y in range(0, 801, 50):
        x_coords = []
        for x in range(-500, 501, 50):
            x_coords.append([x, y, depth])
        coords.append(x_coords)
    coords.reverse()
    return np.array(coords)

print(generate_real_world_coords(2000, 10, 10))

#####

plt.imshow(generate_real_world_coords(:,:,0))
