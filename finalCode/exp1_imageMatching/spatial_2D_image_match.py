import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import time
from spatial_2d_image_match_functions import *

def main():

    # Record start 
    start = time.time()
    
    # Read input files and convert to grayscale
    patternDir = "wallypuzzle_rocket_man.png"
    templateDir = "wallypuzzle_png.png"

    # gray 1
    pattern = read_image( patternDir )
    template = read_image( templateDir )

    pattern_gray = convert_gray( pattern )
    template_gray = convert_gray( template ) 

    # mean shift
    pattern_ms = pattern_gray - np.mean(pattern_gray)
    template_ms = template_gray - np.mean(template_gray)
    # print(pattern_image)
    # Find position of max ccr value, where the pattern image is found in
    # template
    start = time.time()
    image_cross, image_cross_value = find_offset( pattern_ms, template_ms )
    end = time.time()
    
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
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)  
    plt.hist(lum_img_1.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of pattern")
    plt.subplot(1,2,2)  
    plt.hist(lum_img_2.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.title("Pixel intensity of template")
    plt.show()


    #function to find image centre
    vertCen = pattern_gray.shape[1]/2
    horCen = pattern_gray.shape[0]/2

    #plot mark where pattern is found
    plt.imshow( template )  
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
Offset_y =  528 Offset_x =  982 value = 0.520092887633342
<class 'numpy.ndarray'>
Offset_y =  529 Offset_x =  983 Cross value =  0.3724158963277624 run time =  167.4924819469452
"""

