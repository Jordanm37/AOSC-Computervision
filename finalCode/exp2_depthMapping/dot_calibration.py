# importing necessary libraries
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial import distance

np.set_printoptions(precision=2)

mpl.rcParams['figure.figsize'] = (12, 6)

def save_plot(label):   
    plt.savefig(os.path.join("figures","calibration",label + ".png"))
    
def fun1(img_orig, img_template, saved_image_name):
    """
    
    
    Inputs:
    ----------------
        img_orig          BGR Image, image to search for template 
        
		img_template      BGR Image, with similar dimensionality to pattern
        
        saved_image_name  string, name for saved plot 
                
    Output: 
    ----------------
        c22  
        
        euclidian_distance 
     """
     
    img_bgr = img_orig
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Applies the bilateral filter to an image to reduce unwanted noise 
    img_filter = cv2.bilateralFilter(img_rgb, d = 9, sigmaSpace = 75, sigmaColor =75)
    
    # To convert image to gray scale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # To convert image to gray scale after applying filter
    img_gray_smoothed = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY) 
    
    img_bgr_t = img_template
    img_rgb_t = cv2.cvtColor(img_bgr_t, cv2.COLOR_BGR2RGB)
    
    #smooth image
    img_blur_t = cv2.bilateralFilter(img_rgb_t, d = 9, sigmaSpace = 75, sigmaColor =75)
    
    #convert image to gray
    img_gray_t = cv2.cvtColor(img_rgb_t, cv2.COLOR_RGB2GRAY)
    img_gray_t1 = cv2.cvtColor(img_blur_t, cv2.COLOR_RGB2GRAY)
   
    res_L = cv2.matchTemplate(img_gray_smoothed, img_gray_t1, cv2.TM_CCORR_NORMED)
        
    # This parameter (0.99) should be set by trial and error method after visulising 
    threshold_1 = 0.99
    loc = np.where( res_L >= threshold_1)
    pt_p = (0,0)
    c = np.asarray(pt_p)
    c = c.reshape(1,len(c))
    w, h = img_gray_t1.shape[::-1]
    n = 0
    
    # To count number of circle
    for pt in zip(*loc[::-1]):
    
        c1 = np.asarray(pt)
        c1 = c1.reshape(1, len(c1))
        c  = np.append(c, c1, axis = 0)    
        n  = n + 1  
       
    c = np.delete(c, 0, 0)  # first zero for first row and second zero for axis=0 (row)
    
    # To avoid detection of same circle many times
    for i in range(n):
        pt_p = c[i,]
        # for j in range(i+1,n):
        for j in range(i+1, min(i + 1 + int(0.10 * n), n)):
            pt = c[j,]
            euclidian_distance = distance.euclidean(pt_p, pt)
            if euclidian_distance < w:
                c[j,] = 0
                
    c2 = c[~(c==0).all(1)]
    c21 = c2[np.argsort(c2[:, 0])]
    c22 = np.zeros(np.shape(c21))

    for i1 in range(21):  # 21 for no of dot in one row and 17 for no of dot in one column
        C_l0 = c21[(i1 * 17):(i1 * 17 + 17),:]
        c22[(i1 * 17):(i1 * 17 + 17),:] = C_l0[np.argsort(C_l0[:, 1])]
    
    euclidian_distance = distance.euclidean(c22[0,:], c22[len(c22)-1,:])

    img_rgb1 = np.copy(img_rgb)
    # To print and display detected circles
    for l in range(len(c2)):
        cv2.circle(img_rgb1, (int(round(c22[l,0] + w/2)), int(round(c22[l,1] + h/2))), 1, (0,255,255), 50)

    # l = 22
    # cv2.circle(img_rgb1, (int(round(c22[l,0] + w/2)), int(round(c22[l,1] + h/2))), 1, (0,255,255), 50)    
    #print("Circle count by CCORR_NORMED method : ",len(c22))            
    plt.figure()
    plt.subplot(121),plt.imshow(img_orig)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_rgb1)
    plt.title('Output Image by CCORR_NORMED'), plt.xticks([]), plt.yticks([])
    #plt.show()
    save_plot(saved_image_name)
    return c22, euclidian_distance

def reverse(reversed_array):
    temp_reversed_array = np.copy(reversed_array)
    j = 0
    k = 20
    
    for i in range(357):
        if i % 21 ==0:
            j = 0
            k = 20
        temp_reversed_array[i-j+k,:] = reversed_array[i,:]
        j = j + 1
        k = k - 1
    return temp_reversed_array

def fun_predict(circle_left, circle_right, op, z_factor):
    """
    finding coefficient and prediction for each image pair
    
    Inputs:
    ----------------
        circle_left       2d-Array, left rgb image path
        
		circle_right      2d-Array, right rgb image path
        
        op                2d-Array, 
        
        z_factor          float, a fine-tune parameter for calculating final euclidean distance
        
    Output: 
    ----------------
        coef  List, coefficient of determination for x and y
        
        pred 2d-Array, output of regression model and final distance
     """
     
    complete_circle = np.zeros((len(circle_left),4))
    
    final_distance = 0.5 * (distance.euclidean(circle_left[0, :], circle_left[len(circle_left)-1, :]) + \
              (distance.euclidean(circle_right[0, :], circle_right[len(circle_right)-1, :])))
              
    complete_circle[:,0:2]= circle_left
    complete_circle[:,2:4]= circle_right  
    
    ip = PolynomialFeatures(degree=2, include_bias=True).fit_transform(complete_circle) 
    
    model_x = LinearRegression(fit_intercept=False).fit(ip, op[:,0])
    model_y = LinearRegression(fit_intercept=False).fit(ip, op[:,1])
    
    r_sq = model_x.score(ip, op[:,0])
    print('coefficient sequence \n A1, A2, A3, A4, A5, A12, A6, A7, A8, A13, A9, A10, A14, A11, A15')
    
    intercept, coefficients_x = model_x.intercept_, model_x.coef_
    
    print('coefficient of determination for x:', r_sq)
    print('coefficients for x:' + str(coefficients_x))
    
    r_sq = model_y.score(ip, op[:,1])
    intercept, coefficients_y = model_y.intercept_, model_y.coef_
    
    print('coefficient of determination for y', r_sq)
    print('coefficients for y:' + str(coefficients_y))
    
    pred = np.zeros((len(circle_left),3))
    coefficients = np.stack((coefficients_x, coefficients_y), axis=1)
    
    print('depth:' + str(final_distance / z_factor))
    
    pred[:,0] = np.round(model_x.predict(ip))
    pred[:,1] = np.round(model_y.predict(ip))  
    pred[:,2] = np.round(final_distance / z_factor)
    
    return coefficients, pred


def calculate_CoefAndPred(leftImgPath, rightImgPath, img_templatePath, divisionCoef):
    """
    finding coefficient and prediction for each image pair
    
    Inputs:
    ----------------
        leftImgPath       string, left rgb image path
        
		rightImgPath      string, right rgb image path
        
        img_templatePath  string, template path
        
        divisionCoef      float, a fine-tune parameter for calculating final euclidean distance
        
    Output: 
    ----------------
        coef              List, coefficient of determination for x and y
        
        pred              2d-Array, output of regression model and final distance
     """
     
    img_folder = "cal_images"   
    print("\n\n For image:" + str(leftImgPath)) 
    left_img = cv2.imread(os.path.join(img_folder, leftImgPath))
    right_img = cv2.imread(os.path.join(img_folder, rightImgPath))
    img_template = cv2.imread(img_templatePath)
  
    saved_left_image_name = leftImgPath.replace(".png","")
    saved_right_image_name = rightImgPath.replace(".png","")
    
    circle_left, distance_left = fun1(left_img, img_template, saved_left_image_name )
    circle_right, distance_right = fun1(right_img, img_template, saved_right_image_name)  
    
    zFactor = 0.5 * ( distance_left + distance_right ) / divisionCoef
    
    op = np.zeros((len(circle_left),2)) 
    for i in range(21):
      op[i * 17:i * 17 + 17, 0] = -500 + i * 50
      op[i * 17:i * 17 + 17, 1] = np.linspace(800, 0, 17)
      
    coef, pred = fun_predict(circle_left, circle_right, op, zFactor)
    
    return coef, pred

def main():

    #calculate coefficient and prediction for each image       
    templatePath = "T4.tiff" 
    
    coef,pred = calculate_CoefAndPred('cal_image_left_2000.tiff', 'cal_image_right_2000.tiff' , templatePath, 2000)
    coef,pred = calculate_CoefAndPred('cal_image_left_1980.tiff', 'cal_image_right_1980.tiff' , templatePath, 1980)
    coef,pred = calculate_CoefAndPred('cal_image_left_1960.tiff', 'cal_image_right_1940.tiff' , templatePath, 1960)
    coef,pred = calculate_CoefAndPred('cal_image_left_1940.tiff', 'cal_image_right_1940.tiff' , templatePath, 1940)
    coef,pred = calculate_CoefAndPred('cal_image_left_1920.tiff', 'cal_image_right_1920.tiff' , templatePath, 1900)
    coef,pred = calculate_CoefAndPred('cal_image_left_1900.tiff', 'cal_image_right_1900.tiff' , templatePath, 1900)

if __name__ == '__main__':
    main()