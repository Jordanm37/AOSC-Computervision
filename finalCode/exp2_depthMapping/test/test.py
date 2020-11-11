from crossCorrFunctions import *

orientations = ["left", "right"]
distFromCam = [2000,1980,1960,1940,1920,1900]
allCalibIms = filetype_in_dir_w_path("cal_images\\", "tiff")

template_loc = "gauss2D.png"
imLocCom = "Data\\"

gauss2D = genGausTemplate()
cv2.imwrite(template_loc, gauss2D)
plt.title("Gauss template")
plt.contourf(gauss2D)
plt.show()


#Run and save points detected from calibrated image

for calibIm in allCalibIms:
    print(datetime.now())
    ori, dist = calibIm.split('.tiff')[0].split('_')[-2], calibIm.split('.tiff')[0].split('_')[-1]
    print(ori, dist)    
    coords, scores, matchImages = runFindCorr(template_loc, calibIm, thresh=0.95)
    store_output_dict = {}
    store_output_dict['Coordinates'] = coords
    store_output_dict['CorrMtx'] = scores
    pickleFname, _ = storeAsPickle(fname = 'Data\\'+str(ori)+'_'+str(dist)+'.p', data = store_output_dict)
    print(datetime.now())


# for dist in distFromCam:
#     print(datetime.now())
#     print(dist)
#     l_im_loc = imLocCom+"left_"+str(dist)+".tiff"
#     r_im_loc = imLocCom+"right_"+str(dist)+".tiff"
#     l_img = cv2.imread(l_im_loc)
#     r_img = cv2.imread(r_im_loc)    
#     l_XY = findClusterCenters(coordDF = getCoordDF(ori = 'right', dist = dist), display = False)
#     r_XY = findClusterCenters(coordDF = getCoordDF(ori = 'right', dist = dist), display = False)
#     XYD = getObjectPoints(r_no = 17, c_no = 21, spacing = 50, depth = dist)
#     l_XY = np.array(l_XY)
#     XYD = np.array(XYD)

#     imh, imw = l_img.shape[0], l_img.shape[1]
#     l_ret, l_mtx, l_dist, l_rvecs, l_tvecs = cv2.calibrateCamera(XYD, l_XY, (imh, imw),None,None)
#     print('Done')
#     r_ret, r_mtx, r_dist, r_rvecs, r_tvecs = cv2.calibrateCamera(XYD, r_XY, (imh, imw),None,None)
#     print(l_ret, l_mtx, l_dist, l_rvecs, l_tvecs, r_ret, r_mtx, r_dist, r_rvecs, r_tvecs)
#     print('_______________________________________________________________________________')
#     (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(XYD, l_XY, r_XY,
#         l_mtx, l_dist,
#         r_mtx, r_dist,
#         (imh, imw), None, None, None, None,
#         cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
#     print(_, _, _, _, _, rotationMatrix, translationVector, _, _) 
