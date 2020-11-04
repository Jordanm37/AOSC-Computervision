# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:05:04 2020
Last Updated on 23 Oct 20
Combine Trial 1
"""

# importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

'''
Image reading and preprocessing
'''
img_orig = cv2.imread(r'C:\\Users\Jordan\\Documents\\\AOSC-Computervision\\Part_3\\2.png')
img_bgr = img_orig
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# Applies the bilateral filter to an image to reduce unwanted noise 
img_filter = cv2.bilateralFilter(img_rgb, d = 9, sigmaSpace = 75, sigmaColor =75)
# To convert image to gray scale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
# To convert image to gray scale after applying filter
img_gray1 = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY)
# To convert image to gray scale after sharpening
kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
img_gray2 = cv2.filter2D(img_gray, -1, kernel_sharpening)
img_gray3 = cv2.medianBlur(img_gray,1)  # Blurs an image using the median filter.

# To import temlplate as it is required to count circle by CCORR_NORMED method
# and applying same precessing 
imr_template = cv2.imread(r'C:\\Users\Jordan\\Documents\\\AOSC-Computervision\\Part_3\\2_T4.PNG')
img_bgr_t = imr_template
img_rgb_t = cv2.cvtColor(img_bgr_t, cv2.COLOR_BGR2RGB)
img_blur_t = cv2.bilateralFilter(img_rgb_t, d = 9, sigmaSpace = 75, sigmaColor =75)
img_gray_t = cv2.cvtColor(img_rgb_t, cv2.COLOR_RGB2GRAY)
img_gray_t1 = cv2.cvtColor(img_blur_t, cv2.COLOR_RGB2GRAY)
img_gray_t2 = cv2.filter2D(img_gray_t, -1, kernel_sharpening)

## Method 1 of CCORR_NORMED to count circles

res = cv2.matchTemplate(img_gray2,img_gray_t2,cv2.TM_CCORR_NORMED)

# This parameter (0.5-0.95) should be set by trial and error method after visualizing 
threshold_1 = 0.65   
loc = np.where( res >= threshold_1)
pt_p = (0,0)
c = np.asarray(pt_p)
c = c.reshape(1,len(c))
w, h = img_gray_t2.shape[::-1]
n = 0

# To count number of circle
for pt in zip(*loc[::-1]):

    c1 = np.asarray(pt)
    c1 = c1.reshape(1, len(c1))
    c = np.append(c,c1,axis = 0)    
    n = n+1  
   
c = np.delete(c, 0, 0)  # first zero for first row and second zero for axis=0 (row)

# To avoid detection of same circle many times
for i in range(n):
    pt_p = c[i,]
    for j in range(i+1,n):
        pt = c[j,]
        d = distance.euclidean(pt_p, pt)
        if d < w/1.5:
            c[j,] = 0
            
c2 = c[~(c==0).all(1)]
img_rgb1 = np.copy(img_rgb)

# To print and display detected circles
for l in range(len(c2)):
    cv2.circle(img_rgb1, (int(round(c2[l,0] + w/2)), int(round(c2[l,1] + h/2))), 1, (0,255,255), 5)
print("Circle count by CCORR_NORMED method : ",len(c2))            
plt.figure()
plt.subplot(121),plt.imshow(img_orig)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_rgb1)
plt.title('Output Image by CCORR_NORMED'), plt.xticks([]), plt.yticks([])


## Method 2 of Hough circle method to count circles
# minDist, param1 and param2 must be refined by trial and error
circles = cv2.HoughCircles(img_gray3, cv2.HOUGH_GRADIENT, dp = 1, minDist = 10,
                          param1=100, param2=15, minRadius=0, maxRadius=18)
detected_circles = np.uint16(np.around(circles))
img_rgb2 = np.copy(img_rgb)
# To display detected circles
for (x, y ,r) in detected_circles[0, :]:
    # cv2.circle(img_rgb, (x, y), r, (0, 255, 0), 1)
    cv2.circle(img_rgb2, (x, y), 2, (255, 0, 255), 5)
    
count_1 = len(detected_circles[0, :])
print("Circle count by Hough circle method : ",count_1) 

plt.figure()
plt.subplot(121), plt.imshow(img_orig)
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_rgb2)
plt.title('Output by Hough circle method'), plt.xticks([]), plt.yticks([])

## The distribution of Sphere diameter and average in pixels
Radius = circles[0,:,2]
Diameter = Radius*2
mean_Diameter = np.mean(Diameter)
hist1, distribution_Diameter = np.histogram(Diameter, bins=12)
fig1, ax1 = plt.subplots()
ax1.hist(Diameter, distribution_Diameter, cumulative=False)
ax1.set_xlabel('Diameter')
ax1.set_ylabel('Frequency')
plt.title('Distribution of Sphere Diameter')
print("Average Circle diameter (in pixel): ",mean_Diameter)

# Enter scale of the image manually if you want to get diameter in micrometere
# in this example 86 pixel is equal to 3 micrometer
pixel_for_scale = 84
scale_given = 3        # in micrometer
print("Average Circle diameter (in micrometer): ",
      scale_given*mean_Diameter/pixel_for_scale)

## The percentage of the image that is covered with the spheres and not covered
# To identify threshold gor gray to black and white conversion
s0 = np.concatenate(img_gray)
hist4, gray_dist = np.histogram(s0, bins=256)
fig4, ax4 = plt.subplots()
ax4.hist(s0, gray_dist, cumulative=False)
ax4.set_xlabel('Gray scale')
ax4.set_ylabel('Frequency')
plt.title('Distribution of gray value in image')

# Here 60 is threshold
(thresh, img_bw) = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
s1 = np.concatenate(img_bw)
hist2, bw_data = np.histogram(s1, bins=2)
circle_white = hist2[1]
no_circle_black = hist2[0]
# The percentage of the image that is covered with the spheres
circle_percentage = circle_white/(circle_white+no_circle_black)*100 
print("The percentage of the image that is covered with the spheres (in %): ",
      circle_percentage)
# The percentage of the image that is covered with no spheres
no_circle_percentage = no_circle_black/(circle_white+no_circle_black)*100
print("The percentage of the image that is not covered with the spheres (in %): ",
      no_circle_percentage)

fig2, ax2 = plt.subplots()
ax2.hist(s1, bw_data, cumulative=False)
ax2.set_xlabel('Black and White color')
ax2.set_ylabel('Frequency')
plt.title('Distribution of Black and white color in image')


## Distinguish whether the image is 'good' or 'bad'
packing_density = circle_percentage
if packing_density<70:
    print("This image is Bad")
else:
    print("This image is Good")
    
## The distribution of sphere separation
dist = np.zeros((count_1,count_1))
Circle_coordinates = circles[0,:,0:2]   
for i in range(count_1):
    c1 = Circle_coordinates[i,:]
    for j in range(count_1):
        c2 = Circle_coordinates[j,:]
        dist[i,j] = distance.euclidean(c1, c2)
        
dist2 = np.array(dist)
dist2[dist2 > 1.5*mean_Diameter] = 0
cd1 = np.zeros((count_1,1))
for i2 in range(count_1):
    a1 = dist2[i2,:]
    a2 = a1[a1>0]
    cd1[i2,:] =  np.mean(a2)

cd = cd1[~np.isnan(cd1)]
cd1 = np.nan_to_num(cd1)
hist3, ss_dist = np.histogram(cd, bins = 12)    
fig3, ax3 = plt.subplots()
ax3.hist(cd, ss_dist, cumulative=False)
ax3.set_xlabel('Sphere Seperation in pixels')
ax3.set_ylabel('Frequency')
plt.title('Distribution of sphere seperation')

mean_ss = np.mean(cd)
print("Average Sphere Seperation (in pixel): ",mean_ss)
print("Average Sphere Seperation (in micrometer): ",
      scale_given*mean_ss/pixel_for_scale)

## The reflection, R, for a surface in an image
R = np.square((((0.793*(mean_Diameter/mean_ss)**(2))+1)**(3/2)-(1.52))/
              (((0.793*(mean_Diameter/mean_ss)**(2))+1)**(3/2)+(1.52)))

print("The Reflection R (in %): ",R)

## Find the number and size of each uncovered region in 'bad' images

if packing_density<70:
    contours, hierarchy = cv2.findContours(image = img_bw,
                                           mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)
    
    Area_threshold = 0.10*3.14/4*mean_Diameter*mean_Diameter
    
    a4 = np.zeros(len(contours))
    
    contours1=[]
     
    for i4 in range(len(contours)):
        a4[i4] = np.shape(contours[i4])[0]
        if a4[i4]>Area_threshold:
            contours1.append(contours[i4]) 
    
      
    sorted_contours = sorted(contours1, key = cv2.contourArea, reverse = True)
    img_rgb_2 = img_rgb.copy()
    img_uncovered_region = cv2.drawContours(img_rgb_2, sorted_contours, contourIdx = -1, 
                             color = (255, 0, 0), thickness = 2)
    plt.figure(),plt.imshow(img_rgb_2)
    
    print("Number of Contours found = " + str(len(contours1))) 

    hist5, contours_dist = np.histogram(a4, bins = 128)    
    fig5, ax5 = plt.subplots()
    ax5.hist(a4, contours_dist, cumulative=False)
    ax5.set_xlabel('countour area')
    ax5.set_ylabel('Frequency of same area countour')
    plt.title('Distribution of countour area')
    
    
    
## Counts how many times there are double layers (a sphere above the base layer) 
count_dl = 0
index_dl = 0
img_rgb3 = np.copy(img_rgb)
for i3 in range(len(cd1)):
    if cd1[i3,]>0:
        # if cd1[i3,]<Diameter[i3,]:
        # if cd1[i3,]<0.95*mean_ss:
        if cd1[i3,]<int(mean_Diameter):
            count_dl = count_dl + 1
            x = int(circles[0,i3,0])
            y = int(circles[0,i3,1])
            cv2.circle(img_rgb3, (x, y), 2, (0, 255, 255), 5)
            # print(i3)
            
plt.figure()
plt.imshow(img_rgb3,cmap = "gray")
plt.title('Double layer in the image')

print("Count of double layer : ",count_dl)
## Characterise images by size of grains - number of hexagon groups, or triangular groups.
from collections import Counter 

dist3 = np.count_nonzero(dist2, axis=0)
a3 = dict(Counter(dist3))
print("number of Triangle group :",a3.get(3),
      "\n number of Quadrilateral group :",a3.get(4),
      "\n number of Pentagon group :",a3.get(5),
      "\n number of Hexagon group :",a3.get(6),
      "\n number of Heptagon group :",a3.get(7),
      "\n number of Octagon group :",a3.get(8),
      "\n number of Nonagon group :",a3.get(9),
      "\n number of Decagon group :",a3.get(10),
      "\n number of Hendecagon group :",a3.get(11),
      "\n number of Dodecagon group :",a3.get(12))
