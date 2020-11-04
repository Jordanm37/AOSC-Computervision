
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import time
from where_wally_functions import *

def main():

    # Record start 
    start = time.time()
    
    #Read input files and convert to grayscale
    patternDir = "wallypuzzle_rocket_man.png"
    templateDir = "wallypuzzle_png.png"

    image_mean_1 = convert_gray( read_image( patternDir ) )
    image_mean_2 = convert_gray( read_image( templateDir ) )


    print(pattern_image)
    
    # read average of image. Finish by converting to greyscale 

    # Find position of max ccr value, where the pattern image is found in
    # template
    image_cross, image_cross_value = find_offset( image_mean_1, image_mean_2 )
    
    end = time.time()
    
    print( "Offset_x = ", image_cross[0], "Offset_y = ", image_cross[1], "Cross value = ", image_cross_value, "run time = ", end - start  )

    plt.ion()

    # test_plot = template_image[ image_cross[0] : image_cross[0] + pattern_image.shape[0],  image_cross[1] : image_cross[1] + pattern_image.shape[1], : ]  
    # plt.imshow( test_plot )

    #Histogram of colour intensities
    lum_img_1 = image_mean_1[:, :,0]
    lum_img_2 = image_mean_2[:, :,0]
    plt.figure()
    plt.subplot(211)  
    plt.hist(lum_img_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.subplot(212)  
    plt.hist(lum_img_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.show()

if __name__ == '__main__':
    
    main()




"""
Offset_x =  528 Offset_y =  982 value = 0.520092887633342
<class 'numpy.ndarray'>
"""


"""

TO put a red cross once found

ax1.annotate("New, previously unseen!", (160, -35), xytext=(10, 15),
             textcoords="offset points", color='red', size='x-small',
             arrowprops=dict(width=0.5, headwidth=3, headlength=4,
                             fc='k', shrink=0.1));

"""



