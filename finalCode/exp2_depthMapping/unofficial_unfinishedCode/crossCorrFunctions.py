import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import multiprocessing
import sys
from datetime import datetime
import time
import pickle
import json
import glob
from os import listdir
from os.path import isfile, join
import pandas as pd
import cv2

def filetype_in_dir_w_path(dir_path,filetype):
    return glob.glob(dir_path+"*."+filetype)

def storeAsPickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return fname, True

def readFromPickle(fname):
    pickle_obj = open(fname, "rb")
    outData = pickle.load(pickle_obj)
    return outData

def genGausTemplate(sigma = 1.0, mu = 0.0, gridDim = 10):
    x, y = np.meshgrid(np.linspace(-1,1,gridDim), np.linspace(-1,1,gridDim))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def normxcorr2(template, image):
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

    for row in range(0, h_img-h_temp+1):
        for col in range(0, w_img-w_temp+1):
            template = template_norm
            sub_image = image_norm[row:row+h_temp, col:col+w_temp]
            correlation = np.sum(template*sub_image)
            normal = np.sqrt( np.sum(template**2) ) * np.sqrt( np.sum(sub_image**2))
            if normal !=0:
                score = correlation / normal
            else: 
                score = 0
            ncc_matrix[row,col] = score

    return ncc_matrix


def find_matches(template, image, thresh=None):
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
    for row in range(0, h_img-h_temp+1):
        for col in range(0, w_img-w_temp+1):
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
    image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    template = cv2.imread(template_name, cv2.IMREAD_COLOR)
    if thresh == None:
        coords, corrMat, match_image = find_matches(template, image)
        match_images = match_image
    else:
        thresh = float(thresh)
        coords, corrMat, match_images = find_matches(template, image, thresh)
    return coords, corrMat, match_images

def getCoordDF(ori, dist, fileloc = 'Data\\'):
    store_output_dict = readFromPickle(fileloc+ori+'_'+str(dist)+'.p')
    cMat = store_output_dict['CorrMtx']
    coords = store_output_dict['Coordinates']
    coords_x, coords_y = map(list, zip(*coords))
    coordDF = pd.DataFrame([],columns = ["CoordX","CoordY"])
    coordDF["CoordX"] = coords_x
    coordDF["CoordY"] = coords_y
    coordDF = coordDF.sort_values(by = ["CoordX", "CoordY"]).reset_index(drop=True)
    return coordDF

def findClusterCenters(coordDF, display = False):
    allXpos = []
    XposFX = pd.DataFrame(coordDF.CoordX.unique().tolist(),columns=["Xpos"])
    XposFX.loc[(XposFX.Xpos.shift()  < XposFX.Xpos - 30),'group'] = 1
    XposFX['group'] = XposFX['group'].cumsum().ffill().fillna(0)
    allgroups = XposFX.group.unique().tolist()
    for g in allgroups:
        eX = round(XposFX[XposFX.group == g].median())
        allXpos.append(eX.Xpos.astype(int))

    clusterCenterDF = pd.DataFrame([], columns=["CX","CY"])
    for eX in allXpos:
        YposFX = coordDF[coordDF.CoordX == eX]
        YposFX = YposFX.sort_values('CoordY')
        YposFX.loc[(YposFX.CoordY.shift()  < YposFX.CoordY - 50),'group'] = 1
        YposFX['group'] = YposFX['group'].cumsum().ffill().fillna(0)
        manygroups = YposFX.group.unique().tolist()
        for g in manygroups:
            eY = round(YposFX[YposFX.group == g].median())
            clusterCenterDF = clusterCenterDF.append({"CX":eX, "CY":eY.CoordY.astype(int)}, ignore_index=True)
    if display:
        plt.figure(figsize = (20,20))
        plt.scatter(clusterCenterDF.CX, clusterCenterDF.CY)
        plt.plot()
    clusterCenterDF = clusterCenterDF.sort_values(by=["CX","CY"]).reset_index(drop=True)

    return clusterCenterDF

def getObjectPoints(r_no = 17, c_no = 21, spacing = 50, depth = 2000):    
    rvals = list(range(-(int(c_no/2))*spacing, (int(c_no/2)+1)*spacing, spacing))
    cvals = list(range(0,r_no*spacing, spacing))
    objectCoords = pd.DataFrame([], columns=["CX","CY","D"])
    for c in cvals:
        for r in rvals:
            objectCoords = objectCoords.append({"CX":r, "CY":c, "D":depth}, ignore_index=True)
    return objectCoords