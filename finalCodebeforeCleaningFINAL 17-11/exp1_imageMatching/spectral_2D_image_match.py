import numpy as np
import time
from spectral_2D_image_match_functions import *
 
def main():

    patternDir = "wallypuzzle_rocket_man.png"
    templateDir = "wallypuzzle_png.png"

    pattern = read_image(patternDir)
    template = read_image(templateDir)
      
    pattern_gray = convert_gray(pattern )
    template_gray = convert_gray(template)
    
    pattern_s = pattern_gray - np.mean(pattern_gray)
    template_s = template_gray - np.mean(template_gray)
  
    start = time.time()
    image_cross, image_cross_value = find_offset(pattern_s, template_s)
    end = time.time()

    #function to find image centre
    vert_cen = pattern_gray.shape[1] / 2
    hor_cen = pattern_gray.shape[0] / 2

    #centre of pattern
    print("Offset_x_co = ", image_cross[1] + hor_cen, "Offset_y_co = ", image_cross[0] + vert_cen, "value =", image_cross_value)
    #top left corner of pattern image
    print("Offset_x_co = ", image_cross[1], "Offset_y_co = ", image_cross[0], "value =", image_cross_value)
    print("run time = ", end - start )
    
    visualize_results(pattern, pattern_s, template, hor_cen, vert_cen, image_cross)

if __name__ == '__main__':
    main()

