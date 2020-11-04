import os
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy import misc
import random

def CrossCorr_2d(Pattern,Template):
    corr = signal.correlate2d(Pattern, Template, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    return(corr,y,x)

SubPlotRow=2
SubPlotCol=2

ROW = 50
COL = 50
Pattern = []
for r in range(0,ROW):
    temp = []
    for c in range(0,COL):
        temp.append(random.randint(0,100))
    Pattern.append(temp)
    print(Pattern[r])

Temp_ROW = 30
Temp_COL = 30
Template = []
for r in range(0,Temp_ROW):
    temp = []
    for c in range(0,Temp_COL):
        temp.append(random.randint(0,100))
    Template.append(temp)
    print(Template[r])

Corr, y, x = CrossCorr_2d(Pattern,Template)

plt.subplot(SubPlotRow,SubPlotRow,1)
plt.imshow(Pattern, cmap='gray')
plt.title("Pattern")

plt.subplot(SubPlotRow,SubPlotRow,2)
plt.imshow(Template, cmap='gray')
plt.title("Template")

plt.subplot(SubPlotRow,SubPlotRow,3)
plt.imshow(Corr, cmap='gray')
s = "Cross-Correlation of Pattern and Template: Best Correlation at (" + str(x) + ", "+str(y)+")"
plt.title(s)
print("Best Cross-Correlation obtained at location (%d, %d)"%(x,y))
plt.show()


