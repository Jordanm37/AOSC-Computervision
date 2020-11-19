# importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial import distance

mpl.rcParams['figure.figsize'] = (12, 6)

def fun1(img_orig, img_template):
    img_bgr = img_orig
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Applies the bilateral filter to an image to reduce unwanted noise 
    img_filter = cv2.bilateralFilter(img_rgb, d = 9, sigmaSpace = 75, sigmaColor =75)
    # To convert image to gray scale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # To convert image to gray scale after applying filter
    img_gray1 = cv2.cvtColor(img_filter, cv2.COLOR_RGB2GRAY) 
    img_bgr_t = img_template
    img_rgb_t = cv2.cvtColor(img_bgr_t, cv2.COLOR_BGR2RGB)
    img_blur_t = cv2.bilateralFilter(img_rgb_t, d = 9, sigmaSpace = 75, sigmaColor =75)
    img_gray_t = cv2.cvtColor(img_rgb_t, cv2.COLOR_RGB2GRAY)
    img_gray_t1 = cv2.cvtColor(img_blur_t, cv2.COLOR_RGB2GRAY)
   
    res_L = cv2.matchTemplate(img_gray1,img_gray_t1,cv2.TM_CCORR_NORMED)
        
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
        c = np.append(c,c1,axis = 0)    
        n = n+1  
       
    c = np.delete(c, 0, 0)  # first zero for first row and second zero for axis=0 (row)
    
    # To avoid detection of same circle many times
    for i in range(n):
        pt_p = c[i,]
        # for j in range(i+1,n):
        for j in range(i+1,min(i+1+int(0.10*n),n)):
            pt = c[j,]
            d = distance.euclidean(pt_p, pt)
            if d < w:
                c[j,] = 0
                
    c2 = c[~(c==0).all(1)]
    c21 = c2[np.argsort(c2[:, 0])]
    c22 = np.zeros(np.shape(c21))

    for i1 in range(21):  # 21 for no of dot in one row and 17 for no of dot in one column
        C_l0 = c21[i1*17:i1*17+17,:]
        c22[i1*17:i1*17+17,:] = C_l0[np.argsort(C_l0[:, 1])]
    
    d = distance.euclidean(c22[0,:], c22[len(c22)-1,:])

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
    return [c22, d]

def reverse(C_r):
    C_r2 = np.copy(C_r)
    j = 0
    k = 20
    for i in range(357):
        if i%21==0:
            j = 0
            k = 20
        C_r2[i-j+k,:] = C_r[i,:]
        j = j + 1
        k = k - 1
    return C_r2


def fun_predict(C_l,C_r,op,z_factor):
    C = np.zeros((len(C_l),4))
    d = 0.5*(distance.euclidean(C_l[0,:], C_l[len(C_l)-1,:])+(distance.euclidean(C_r[0,:], C_r[len(C_r)-1,:])))
    C[:,0:2]= C_l
    C[:,2:4]= C_r  
    ip = PolynomialFeatures(degree=2, include_bias=True).fit_transform(C) 
    model_x = LinearRegression(fit_intercept=False).fit(ip, op[:,0])
    model_y = LinearRegression(fit_intercept=False).fit(ip, op[:,1])
    r_sq = model_x.score(ip, op[:,0])
    print('coefficient sequence \n A1, A2, A3, A4, A5, A12, A6, A7, A8, A13, A9, A10, A14, A11, A15')
    intercept, coefficients_x = model_x.intercept_, model_x.coef_
    print('coefficient of determination for x:', r_sq)
    print('coefficients for x:', coefficients_x, sep='\n')
    r_sq = model_y.score(ip, op[:,1])
    intercept, coefficients_y = model_y.intercept_, model_y.coef_
    print('coefficient of determination for y', r_sq)
    print('coefficients for y:', coefficients_y, sep='\n')
    pred = np.zeros((len(C_l),3))
    coefficients = np.stack((coefficients_x, coefficients_y), axis=1)
    print('depth:', d/z_factor, sep='\n')
    
    pred[:,0] = np.round(model_x.predict(ip))
    pred[:,1] = np.round(model_y.predict(ip))  
    pred[:,2] = np.round(d/z_factor)
    return coefficients, pred

def main():
    orientations = ["left", "right"]
    distFromCam = [2000,1980,1960,1940,1920,1900]
    distFromCamstr = ["2000", "1980", "1960", "1940", "1920", "1900"]
    # print(d)
    # for k,v in d:
    #     print(k, 'corresponds to', v)
    #     print(type(k))
    img_template = cv2.imread('T4.tiff')

    im_orig_l = np.zeros( ( 6,1) )
    im_orig_r =np.zeros( ( 6,1) )
    C_l =np.zeros( ( 6,1) )
    C_r = np.zeros( ( 6,1) )
    d_l =np.zeros( ( 6,1) )
    d_r =np.zeros( ( 6,1) )

    for dist in distFromCam: 
        i = 0
        img_orig_l = cv2.imread('cal_images\\cal_image_left_' + str(dist) + '.tiff')
        img_orig_r = cv2.imread('cal_images\\cal_image_right_' + str(dist) + '.tiff')
        C_l[i], d_l = fun1(img_orig_l, img_template)
        C_r[i], d_r = fun1(img_orig_r, img_template)
        z_factor[i] = 0.5*(d_l+d_r)/(dist) 
        i +=1


    op = np.zeros((len(C_l[0]),2))
                
    for i in range(21):
        op[i*17:i*17+17,0] = -500+i*50
        op[i*17:i*17+17,1] = np.linspace(800,0,17)

    coefficients = []
    Prediction = []
    for dist in distFromCam: 
        i = 0
        print('\n\nfor image_' + str(dist) )
        coefficients[i], Prediction[i] = fun_predict(C_l[i],C_r[i],op,z_factor[i])
        i += 1

if __name__ == '__main__':
    main()



