import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
display_default = True

def files_w_extn(dir_p, ext_n):
    file_names = []
    for file in os.listdir(dir_p):
        if file.endswith(ext_n):
            file_names.append(os.path.join(dir_p, file))    
    return file_names


testim_loc ='AllImages\\test\\' 
alltestimages = files_w_extn(testim_loc, 'png')
print(len(alltestimages))