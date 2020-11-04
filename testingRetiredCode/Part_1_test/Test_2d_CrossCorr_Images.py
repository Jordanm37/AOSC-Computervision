import os
import math
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
from PIL import Image, ImageOps

from scipy import signal
from scipy import misc
import random

def CrossCorr_2d(Pattern,Template):
    corr = signal.correlate2d(Pattern, Template, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    return(corr,y,x)

TODO: #convert and put this into the working version of 

def rgb2gray(imgpath):
    img = Image.open(imgpath)
    pixels = img.load()  
    img_width, img_height = img.size
    Gray = []
    Gray1=[]
    for i in range(0,img_height ):
        temp = []
        for j in range(0,img_width):
            r, g, b = pixels[i, j]
            tr = int(r*0.2989)
            tg = int(g*0.5870)
            tb = int(b*0.1140)
            temp.append(tr+tg+tb)
        Gray.append(temp)    
    for i in range(0,img_height ):
        temp = []
        for j in range(0,img_width):
            temp.append(Gray[j][i])
        Gray1.append(temp)    
    return(Gray1)

SubPlotRow=2
SubPlotCol=2

ProjectPath = os.getcwd()
ImagePath = ProjectPath + "\\Part 1 testing\\Leena.jpg"

Pattern = Image.open(ImagePath)
Gray = rgb2gray(ImagePath)

Temp_ROW = 50
Temp_COL = 50
Template = []
for r in range(0,Temp_ROW):
    temp = []
    for c in range(0,Temp_COL):
        temp.append(random.randint(0,100))
    Template.append(temp)

Corr, y, x = CrossCorr_2d(Gray,Template)

plt.subplot(SubPlotRow,SubPlotRow,1)
plt.imshow(Pattern, cmap='gray')
plt.title("Pattern")

plt.subplot(SubPlotRow,SubPlotRow,2)
plt.imshow(Gray, cmap='gray')
plt.title("Pattern")

plt.subplot(SubPlotRow,SubPlotRow,3)
plt.imshow(Corr, cmap='gray')
s = "Cross-Correlation of Pattern and Template: Best Correlation at (" + str(x) + ", "+str(y)+")"
plt.title(s)
print("Best Cross-Correlation obtained at location (%d, %d)"%(x,y))
plt.show()
