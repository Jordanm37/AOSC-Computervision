# Library imports
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

display_default = False

def files_w_extn(dir_p, ext_n):
    """
    Get path of test and template images directory
 
    Inputs:
    ----------------
        dir_P   the directory where files are being searched

        ext_n   the extension of the files
    
    Output:
    ----------------
        file_names  file names in a directory with particular extension
        
    """
    
    file_names = []
    for file in os.listdir(dir_p):
        if file.endswith(ext_n):
            file_names.append(os.path.join(dir_p, file))    
    return file_names
  
def readAndProcess(imLoc, display=display_default):
    """
    Read and Process the images 
 
    Inputs:
    ----------------
        imLoc     Image Location

        display   default_display
    
    Output:
    ----------------
        rgb, gry, gry_filtered, gry_blurred, img_orig     Display plots of Original Image, rgb image,grey image, gry_filtered image, gry_blurred image, 
        
    """
    
    img_orig = cv2.imread(imLoc)
    rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    gry = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    
    gry_filtered = cv2.filter2D(gry, -1, kernel_sharpening)
    gry_blurred = cv2.medianBlur(gry,1) 
    
    if display:
        plt.imshow(img_orig)
        plt.show()
        
    return rgb, gry, gry_filtered, gry_blurred, img_orig

def getCirclesCN(locs, rgb_testim, ori_im, display=display_default):
    """
    Draw and Display circles of original image after applying CCORR_NORMED Method 
 
    Inputs:
    ----------------
        locs         Location of the desired image

        rgb_testim   Location of rgb test image
  
        display      default display
    
    Output:
    ----------------
        rgb, gry, gry_filtered, gry_blurred, img_orig     Display plots of Original Image, rgb image,grey image, gry_filtered image, gry_blurred image, 
    
    """
    
    # print(len(locs))
    
    c = np.asarray((0,0))
    c = c.reshape(1,len(c))
    w, h = tl_gFltrd.shape[::-1]
    n = 0

    for pt in zip(*locs[::-1]):
        c1 = np.asarray(pt)
        c1 = c1.reshape(1, len(c1))
        c = np.append(c,c1,axis = 0)    
        n = n+1 
        # print(pt, n)
        
    c = np.delete(c, 0, 0)
    
    # print("first for ended")
    
    for i in range(n):
        pt_p = c[i,]
        for j in range(i+1,n):
            pt = c[j,]
            d = distance.euclidean(pt_p, pt)
            if d < w/1.5:
                c[j,] = 0
    # print("2nd for ended")
    
    circles = c[~(c==0).all(1)]
    rgb1 = np.copy(rgb_testim)
    
    for i in range(len(circles)):
        cv2.circle(rgb1, (int(round(circles[i,0] + w/2)), int(round(circles[i,1] + h/2))), 1, (0,255,255), 5)
        
    if display:
        plt.figure()
        plt.subplot(121),plt.imshow(ori_im)
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(rgb1)
        plt.title("Output Image by CCORR_NORMED"), plt.xticks([]), plt.yticks([])
        plt.show()    

    return [circles], rgb1

def getCirclesHC(img_orig, rgb, greyBlurd, min_dist, param_1, param_2, display=display_default):
    """
    Draw and Display circles after applying Hough circle Method 
 
    Inputs:
    ----------------
        img_orig        Original Image

        rgb             Transformed rgb image

        greyBlurd       Transformed blur Image

        min_dist        10

        param_1         100
        
        param_2         15

        display         default display
    
    Output:
    ----------------
        circles, rgb2     Display plots of Original Image, rgb2 image

    """
   
    circles = cv2.HoughCircles(greyBlurd, cv2.HOUGH_GRADIENT, dp = 1, minDist = min_dist,
                          param1=param_1, param2=param_2, minRadius=0, maxRadius=18)
                          
    detected_circles = np.uint16(np.around(circles))
    rgb2 = np.copy(rgb)
    
    for (x, y ,r) in detected_circles[0, :]:
        cv2.circle(rgb2, (x, y), 2, (255, 0, 255), 5)
        
    count_1 = len(detected_circles[0, :])
    
    if display:
        plt.figure()
        plt.subplot(121), plt.imshow(img_orig)
        plt.title("original"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(rgb2)
        plt.title("Output by Hough circle method"), plt.xticks([]), plt.yticks([])
        plt.show()

    return circles, rgb2
    
def avgCircleDiameter(scale_given, pixel_for_scale, circles, display=display_default):
    """
    Get Average circle diameter in pixel and micrometer, detected using hough circle method 
 
    Inputs:
    ----------------
        scale_given         3

        pixel_for_scale     84

        circles             calculated by Hough Circle Method

        display             default display
     
    Output:
    ----------------
        pixel_Diameter, actual_Diameter     Generate the pixel for scale diameter and original diameter of the circle
    
    """
    
    Radius = circles[0,:,2]
    Diameter = Radius * 2
    pixel_Diameter = np.mean(Diameter)
    
    if display:
        hist1, distribution_Diameter = np.histogram(Diameter, bins=12)
        fig1, ax1 = plt.subplots()
        ax1.hist(Diameter, distribution_Diameter, cumulative=False)
        ax1.set_xlabel("Diameter")
        ax1.set_ylabel("Frequency")
        plt.title("Distribution of Sphere Diameter")
        plt.show()
        
    actual_Diameter = ( scale_given * pixel_Diameter ) / pixel_for_scale
    
    return pixel_Diameter, actual_Diameter

def circleCoveredPercentage(gry,display=display_default):
    """
    Get the circles percentage covered or not covered the transformed grey scale image 
 
    Inputs:
    ----------------
        gry            Transformed grey scale image  

        display        default display
     
    Output:
    ----------------
        circle_percentage, no_circle_percentage, img_bw     Percentages of the circle covered or not covered the transformed grey image. 
    
    """
    
    s0 = np.concatenate(gry)
    hist4, gray_dist = np.histogram(s0, bins=256)
    (thresh, img_bw) = cv2.threshold(gry, 60, 255, cv2.THRESH_BINARY)
    
    s1 = np.concatenate(img_bw)
    hist2, bw_data = np.histogram(s1, bins=2)
    
    circle_white = hist2[1]
    no_circle_black = hist2[0]
    
    circle_percentage = circle_white / ( circle_white+no_circle_black ) * 100 
    no_circle_percentage = no_circle_black / ( circle_white+no_circle_black ) * 100

    if display:
        fig4, ax4 = plt.subplots()
        ax4.hist(s0, gray_dist, cumulative=False)
        ax4.set_xlabel("Gray scale")
        ax4.set_ylabel("Frequency")
        plt.title("Distribution of gray value in image")
        fig2, ax2 = plt.subplots()
        ax2.hist(s1, bw_data, cumulative=False)
        ax2.set_xlabel("Black and White color")
        ax2.set_ylabel("Frequency")
        plt.title("Distribution of Black and white color in image")
        plt.show()
        
    return circle_percentage, no_circle_percentage, img_bw

def imageGoodOrBad(percent,thresh):
    """
    Check whether Resulted Image is good or bad 
 
    Inputs:
    ----------------
        percent         The circle value percentage of the transforemd image  

        thresh          70 is considered as threshold value
     
    Output:
    ----------------
       im_qual          Based on the image percentage returrns the quality
   
    """
    
    if percent > thresh:
        im_qual = "good"
    else:
        im_qual = "bad"
        
    return im_qual

def sphereSepReflct(scale_given, pixel_for_scale, numOfCircles, circles, meanDiameter, display=display_default):
    """
    Calculate the average sphere seperation and reflection percentages 
 
    Inputs:
    ----------------
        scale_given         3 as used in above function

        pixel_for_scale     84 as used in above function

        numOfCircles        length of Hough circles at start position 

        circles             numner of Hough circles

        meanDiameter        circlesHCDiameterPixel

        display             default
        
    Output:
    ----------------
       avgSphereSepPixel, avgSphereSepMM, round(reflection*100,2), circleDiameters, distances   returrns the average sphere separation  in pixel and mmicrometer
    
    """
    
    dist = np.zeros(( numOfCircles, numOfCircles ))
    circle_coordinates = circles[0,:,0:2]   
    
    for i in range(numOfCircles):
        c1 = circle_coordinates[i,:]
        for j in range(numOfCircles):
            c2 = circle_coordinates[j,:]
            dist[i,j] = distance.euclidean(c1, c2)
            
    distances = np.array(dist)
    distances[distances > 1.5 * meanDiameter] = 0
    circleDiameters = np.zeros(( numOfCircles , 1 ))
    
    for i2 in range(numOfCircles):
        a1 = distances[i2, :]
        a2 = a1[ a1 > 0 ]
        circleDiameters[i2, :] =  np.mean(a2)
        
    cd = circleDiameters[~np.isnan(circleDiameters)]
    circleDiameters = np.nan_to_num(circleDiameters)
    
    hist3, ss_dist = np.histogram(cd, bins = 12)    
    avgSphereSepPixel = np.mean(cd)
    avgSphereSepMM = ((scale_given * avgSphereSepPixel) / pixel_for_scale)
    
    reflection = np.square((((0.793 * (meanDiameter / avgSphereSepPixel) ** (2)) + 1) ** (3.0/2.0)- (1.52)) /
                (((0.793 * ( meanDiameter / avgSphereSepPixel ) ** (2)) + 1) ** (3.0/2.0) + (1.52)))
                
    if display:
        fig3, ax3 = plt.subplots()
        ax3.hist(cd, ss_dist, cumulative=False)
        ax3.set_xlabel("Sphere Seperation in pixels")
        ax3.set_ylabel("Frequency")
        plt.title("Distribution of sphere seperation")
        plt.show()
        
    return avgSphereSepPixel, avgSphereSepMM, round(reflection * 100, 2), circleDiameters, distances

def sizeOfUncoveredRegion(rgb, mean_Diameter, img_bw, display=display_default):
    """
    Find the size of uncovered region 
 
    Inputs:
    ----------------
        rgb               Transformed rgb image
        
        mean_Diameter     circlesHCDiameterPixel

        img_bw            Grey image

        display           default
        
    Output:
    ----------------
       countour_list   returns the contour list
       
    """
    
    #contours, hierarchy = cv2.findContours(
    
    #note first return is image, changed because new version
    _, contours, hierarchy = cv2.findContours( 
        image = img_bw,
        mode = cv2.RETR_TREE,
        method = cv2.CHAIN_APPROX_SIMPLE)
        
    area_threshold = 0.10 * (3.14 / 4) * mean_Diameter * mean_Diameter
    
    a4 = np.zeros(len(contours))
    countour_list=[]
    
    for i4 in range(len(contours)):
        a4[i4] = np.shape(contours[i4])[0]
        
        if a4[i4]>area_threshold:
            countour_list.append(contours[i4]) 
            
    sorted_contours = sorted(countour_list, key = cv2.contourArea, reverse = True)
    img_uncovered_region = cv2.drawContours(rgb, sorted_contours, contourIdx = -1, 
                            color = (255, 0, 0), thickness = 2)
                            
    hist5, contours_dist = np.histogram(a4, bins = 128)  
    
    if display:
        fig5, ax5 = plt.subplots()
        ax5.hist(a4, contours_dist, cumulative=False)
        ax5.set_xlabel("countour area")
        ax5.set_ylabel("Frequency of same area countour")
        plt.title("Distribution of countour area")
        plt.show()
        
    return countour_list

def numDoubleLayers(dist, circles, rgb, mean_Diameter, circleDiameters, display=display_default):
    """
    Count the double layers 
 
    Inputs:
    ----------------
        dist               array of all the distances
        
        circles            Hough Circles

        rgb                Transformed rgb

        mean_Diameter      Hough Circles Diameter Pixel

        circleDiameters    meanDiameter

        display            default
        
    Output:
    ----------------
       countour_list   returns the contour list

    """
    
    count_dl = 0
    index_dl = 0
    
    for i3 in range(len(circleDiameters)):   
        if circleDiameters[i3,] > 0:
            if circleDiameters[i3,]<int(mean_Diameter):
                count_dl = count_dl + 1
                x = int(circles[0, i3, 0])
                y = int(circles[0, i3, 1])
                cv2.circle(rgb, (x, y), 2, (0, 255, 255), 5)
                
    dist3 = np.count_nonzero(dist, axis=0)
    allcounts = dict(Counter(dist3))
    
    if display:
        plt.figure()
        plt.imshow(rgb,cmap = "gray")
        plt.title("Double layer in the image")
        plt.show()
        
    return allcounts, count_dl

def circleCompPlot(testName, rgb, gry, gFltrd, gblrd, circlesCN, circlesHC):
    """
    Print comparison plots: Circles detected by both techniques
 
    Inputs:
    ----------------
        rgb            Transformed rgb
        
        gry            Transformed gry

        gFltrd         Transformed gFltrd

        gblrd          Transformed gblrd

        circlesCN      for plot CN Cirlces

        circlesHC      for plot HC Cirlces
        
    Output:
    ----------------
       Plots of RGB, Grey, 2D Filtered, Blurred, CCORD_NORMED, Hough circle

    """
    
    label = testName
    plt.figure(figsize=(20,20))
    plt.subplot(321), plt.imshow(rgb)
    plt.title("RGB"), plt.xticks([]), plt.yticks([])
    
    # plot_save(f'RGB_{label}') 
    plt.subplot(322), plt.imshow(gry)
    plt.title("Grey"), plt.xticks([]), plt.yticks([])
    
    # plot_save(f'Grey_{label}')
    plt.subplot(323), plt.imshow(gFltrd)
    plt.title("2D Filtered"), plt.xticks([]), plt.yticks([])
    
    # plot_save(f'2D Filtered_{label}')
    plt.subplot(324), plt.imshow(gblrd)
    plt.title("Blurred"), plt.xticks([]), plt.yticks([])
    
    # plot_save(f'Blurred_{label}')
    plt.subplot(325), plt.imshow(circlesCN)
    plt.title("CCORD_NORMED"), plt.xticks([]), plt.yticks([])
    
    # plot_save(f'CCORD_NORMED_{label}')
    plt.subplot(326), plt.imshow(circlesHC)
    plt.title("Hough circle"), plt.xticks([]), plt.yticks([])
    
    # plot_save(f'Hough circle_{label}')
    #plot_save(f'Comparison_{label}')
    plt.show()

def plot_save(label):
    plt.tight_layout()
    path = os.path.join("..","figures","applicationMotheye","fig_" + label)
    plt.savefig(path,dpi = 250)

#Main execution block
if __name__ == "__main__":
    #Test and template image directories
    testim_loc ="test\\" 
    tmplim_loc ="templates\\"

    #Sample test and template image path
    # testImage = "test\\testImages_1.png"
    # tmplImage = "templates\\template_1.jpeg"

    #Get path for all test and template image
    alltestimages = files_w_extn(testim_loc, "png")
    alltmplimages = files_w_extn(tmplim_loc, "jpeg")
    
    print(len(alltmplimages))
    print(len(alltestimages))
    
    #Start processing each image pairs
    for idx in range(0, len(alltestimages)):

        #Print start of the processing step
        print("______________________________________________________________")
        print("Processing image set "+str(idx+1))
        testImage = alltestimages[idx]
        tmplImage = alltmplimages[idx]
        print("Input image names :: ")
        print(testImage, tmplImage)
        print("______________________________________________________________")

        #Transformation applied on input image: Color, Filter and Blur 
        tt_rgb, tt_gry, tt_gFltrd, tt_gblrd, tt_ori = readAndProcess(testImage)
        tl_rgb, tl_gry, tl_gFltrd, tl_gblrd, tl_ori = readAndProcess(tmplImage)


        thresh = 0.65
        res = cv2.matchTemplate(tt_gFltrd, tl_gFltrd, cv2.TM_CCORR_NORMED)


        locs = np.where(res >= thresh)    

        #Get circles using method : CCORR_NORMED
        circlesCN, plotCNCirlces = getCirclesCN(locs, tt_rgb, tt_ori, display=display_default)
        print("Number of circles detected by CCORR_NORMED method :: "+str(len(circlesCN[0])))

        #Get circles using method : Hough circles
        circlesHC, plotHCCirlces = getCirclesHC(img_orig=tt_ori, rgb = tt_rgb,
        greyBlurd=tt_gblrd, min_dist=10, param_1=100, param_2=15, display=display_default)
        print("Number of circles detected by CCORR_NORMED method :: "+str(len(circlesHC[0])))

        #Calculate average circle diameter
        circlesHCDiameterPixel, circlesHCDiameterActual = avgCircleDiameter(scale_given=3, \
        pixel_for_scale = 84, circles=circlesHC, display=display_default)

        print("Average circle diameter in pixel and micrometer, detected using hough circle method :: ", \
        str(circlesHCDiameterPixel), str(round(circlesHCDiameterActual,3)))

        #Calculate percentage of image covered with circles
        circleP, noCircleP, imgBW = circleCoveredPercentage(gry = tt_gry, display=display_default)
        print("Percentage Covered/Not covered in circles :: ", str(circleP), str(noCircleP))

        #Find average sphere seperation and reflection percentages
        avgSphereSepPixel, avgSphereSepMM, reflection, cds, distances = sphereSepReflct(scale_given=3,\
        pixel_for_scale=84, numOfCircles=len(circlesHC[0]), circles=circlesHC, meanDiameter=circlesHCDiameterPixel,\
        display=display_default)

        print("Average Sphere Seperation (in pixel) :: ", avgSphereSepPixel)
        print("Average Sphere Seperation (in micrometer) :: ", avgSphereSepMM)        
        print("The reflection in % :: ", reflection)

        #Find if image is good or bad       
        print("This image is "+imageGoodOrBad(circleP, 70))

        #Find the size of uncovered region
        contour_list = sizeOfUncoveredRegion(rgb = tt_rgb, mean_Diameter=circlesHCDiameterPixel, \
        img_bw=imgBW, display=display_default)
        print("Number of Contours found :: " + str(len(contour_list)))

        allcounts, dlCount = numDoubleLayers(dist = distances, circles=circlesHC, rgb=tt_rgb,\
        mean_Diameter=circlesHCDiameterPixel, circleDiameters=cds,display=display_default)

        print("Count of double layer :: ",dlCount) 
        print("number of Triangle group :",allcounts.get(3),
        "\n number of Quadrilateral group :",allcounts.get(4),
        "\n number of Pentagon group :",allcounts.get(5),
        "\n number of Hexagon group :",allcounts.get(6),
        "\n number of Heptagon group :",allcounts.get(7),
        "\n number of Octagon group :",allcounts.get(8),
        "\n number of Nonagon group :",allcounts.get(9),
        "\n number of Decagon group :",allcounts.get(10),
        "\n number of Hendecagon group :",allcounts.get(11),
        "\n number of Dodecagon group :",allcounts.get(12))
        print("______________________________________________________________")

        #Print comparison plots: Circles detected by both techniques
        circleCompPlot((testImage), rgb=tt_rgb, gry=tt_gry, gFltrd=tt_gFltrd,\
        gblrd=tt_gblrd, circlesCN = plotCNCirlces, circlesHC=plotHCCirlces )
