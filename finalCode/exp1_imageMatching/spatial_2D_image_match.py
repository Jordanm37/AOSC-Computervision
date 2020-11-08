import time
from spatial_2D_image_match_functions import *

def main():

    # Record start 
    start = time.time()
    
    # Read input files and convert to grayscale
    patternDir = "wallypuzzle_rocket_man.png"
    templateDir = "wallypuzzle_png.png"

    # gray 1
    pattern = read_image(patternDir)
    template = read_image(templateDir)

    pattern_gray = convert_gray(pattern)
    template_gray = convert_gray(template) 

    # mean shift
    pattern_ms = pattern_gray - np.mean(pattern_gray)
    template_ms = template_gray - np.mean(template_gray)

    # print(pattern_image)
    # Find position of max ccr value, where the pattern image is found in
    # template
    start = time.time()
    image_cross, image_cross_value = find_offset(pattern_ms, template_ms)
    end = time.time()

    #function to find image centre
    vertCen = pattern_gray.shape[1] / 2
    horCen = pattern_gray.shape[0] / 2

    #centre of pattern
    print("Offset_x_co = ", image_cross[1] + horCen, "Offset_y_co = ", image_cross[0] + vertCen, "value =", image_cross_value)
    #top left corner of pattern image
    print("Offset_x_co = ", image_cross[1], "Offset_y_co = ", image_cross[0], "value =", image_cross_value)
    print("run time = ", end - start )
    
    #plots of original, greyscale, intensity and mark where pattern is found
    visualize_results(pattern, pattern_ms, template, template_ms, horCen, vertCen, image_cross)

if __name__ == '__main__':  
    main()

"""
Offset_y =  528 Offset_x =  982 value = 0.520092887633342
<class 'numpy.ndarray'>
Offset_y =  529 Offset_x =  983 Cross value =  0.3724158963277624 run time =  167.4924819469452
"""

