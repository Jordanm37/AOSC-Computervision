import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import pandas as pd
import cv2
import os

def normxcorr2(template, image):
    """
    Normalisation cross correlation functions
    
    Inputs:
    ----------------
        template   GrayScale Image, with similar dimensionality to pattern
        
		image      GrayScale Image, image to search for template
        
    Output: 
    ----------------
        norm      2D-Array,  normalised cross-correlation matrice
     """
     
    template_gray = template
    image_gray = image
    h_temp, w_temp = template_gray.shape
    h_img, w_img = image_gray.shape

    min_temp = 0
    max_temp = 0
    
    for i in range(0, h_temp):
        for j in range(0, w_temp):        
            if template_gray[i][j] > max_temp:
                max_temp = template_gray[i][j]               
            if template_gray[i][j] < min_temp:
                min_temp = template_gray[i][j]

    min_img = 0
    max_img = 0
    
    for i in range(0, h_img):
        for j in range(0, w_img):
            if image_gray[i][j] > max_img:
                max_img = image_gray[i][j]
            if image_gray[i][j] < min_temp:
                min_temp = image_gray[i][j]
                
    if (max_temp - min_temp) != 0 :            
        template_norm = (template_gray - min_temp) / (max_temp - min_temp)
    else:
        template_norm = 0
    if (max_img - min_img) != 0 :
        image_norm = (image_gray - min_img) / (max_img - min_img)
    else:
        image_norm = np.zeros(image.shape)
    ncc_matrix = np.zeros((h_img-h_temp+1, w_img-w_temp+1))

    for row in range(0, h_img - h_temp + 1):
        for col in range(0, w_img - w_temp + 1):
            template = template_norm
            sub_image = image_norm[row:(row + h_temp), col:(col + w_temp)]
            correlation = np.sum(template * sub_image)
            normal = np.sqrt(np.sum(template**2)) * np.sqrt(np.sum(sub_image**2))
            if normal !=0:
                score = correlation / normal
            else: 
                score = 0
            ncc_matrix[row,col] = score

    return ncc_matrix

def find_matches(template, image, thresh=None):
    """
    finding matches between template and image function
    
    Inputs:
    ----------------
        template   RGB Image, with similar dimensionality to pattern
        
		image      RGB Image, image to search for template
        
    Output: 
    ----------------
        coordinates     List,  corr_matrix (x,y) indices that corr_matrix[x,y] > thresh
        
        corr_matrix     2D-Array,  normalised cross-correlation matrice
        
        match_images    List,  matched images with rectangle
     """
     
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corr_matrix = normxcorr2(template_gray, image_gray)
    
    image_copy = image_gray
    coordinates = []
    match_images = []
    max_val = 0
    (max_y, max_x) = (0, 0)

    h_temp, w_temp = template_gray.shape
    h_img, w_img = image_gray.shape

    color = (0,255,0)
    thickness = 2

    if thresh:
        max_val = thresh
    else:
        max_val = 0
        
    for row in range(0, h_img-h_temp + 1):
        for col in range(0, w_img-w_temp + 1):
            if thresh:
                if corr_matrix[row,col] > thresh:
                    (max_y, max_x) = row, col
                    coordinates.append((max_x,max_y))
                    start_point = (max_x,max_y)
                    end_point = (max_x+w_temp, max_y+h_temp)
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                    match_images.append(image)
                    image_gray = image_copy
            else:
                if corr_matrix[row,col] > max_val:
                    (max_y, max_x) = row, col
                    max_val = corr_matrix[row,col]
                    
    if thresh == None:
        start_point = (max_x, max_y)
        end_point = (max_x + w_temp, max_y + h_temp)
        coords = (max_x, max_y)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        return coords, corr_matrix, image

    return coordinates, corr_matrix, match_images 

def runFindCorr(template_name, image_name, thresh=None):
    """
    finding correlation function
    
    Inputs:
    ----------------
        template_name   string, template image path
        
		image_name      string, image path to search for template
        
        thresh          float, level of cross-correlation
        
    Output: 
    ----------------
        coords     List,  corr_matrix (x,y) indices that corr_matrix[x,y] > thresh
        
        corrMat     2D-Array,  normalised cross-correlation matrice
        
        match_images    List,  matched images with rectangle
     """
     
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    template = cv2.imread(template_name, cv2.IMREAD_COLOR)
    
    if thresh == None:
        coords, corrMat, match_image = find_matches(template, image)
        match_images = match_image
    else:
        thresh = float(thresh)
        coords, corrMat, match_images = find_matches(template, image, thresh)
    return coords, corrMat, match_images
