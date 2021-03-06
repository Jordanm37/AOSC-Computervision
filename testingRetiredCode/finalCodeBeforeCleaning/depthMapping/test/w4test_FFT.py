import glob
from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time
from FT_functions import *
from testcorr import *

def load_images(filepath):
    '''
    
    This  Function reads two images in the folder; one 
    with '_right' in filename another in '_left' in filename. 
    Convert the two images to numpy array and convert 
    from rgb to blackand white and return the two images 
    as pattern and template
    Return false if there is no file with 
    '_right' or '_left' in the given folder
    '''
    
    file_ext = "*.tiff*"
    template_fn = ""
    pattern_fn = ""
    
    fileNames = glob.glob(filepath + file_ext)

    for fileName in fileNames:
        print(fileName)
        
        # right_ is for the file type in the test folder may be removed if files are named properly
        if "right_portal" in fileName:  
            template_fn = fileName
            
        elif "left_portal" in fileName:
            pattern_fn = fileName

    if template_fn == "" or pattern_fn == "":
        print("Both (or one) files not available")
        return False
        
    try:
        template = np.array(Image.open(template_fn))
        pattern = np.array(Image.open(pattern_fn))
    except expression as identifier:
        print("Error in Reading the files")
        return False
        
    try:
        template_gray = 0.2989 * template[:, :, 0] \
                      + 0.5870 * template[:, :, 0] \
                      + 0.1140 * template[:, :, 0]
                   
                   
        pattern_gray = 0.2989 * pattern[:, :, 0] \
                     + 0.5870 * pattern[:, :, 0] \
                     + 0.1140 * pattern[:, :, 0]
    except expression as identifier:
        print("Error in Converting from RGB to Gray scale")
        return False

    return pattern_gray, template_gray

def get_grid(image, numY, numX):
    '''
    
    This  Function receives one image and divides the image 
    into multiple windows based on numY and numX as division 
    of widows in y and x axis 
    This returns a 2D array of smaller images(windows)
    '''
    
    #import pdb; pdb.set_trace()
    height, width = image.shape
    
    #extra size +1 drops off if present becasue of integer division
    len_y = int(height / numY) 
    len_x = int(width / numX) 

    wins = []
    for i in range(0, numY):
        row = []
        for j in range(0, numX):
            row.append(image[(i * len_y):(((i + 1) * len_y)),
                             (j * len_x):(((j + 1) * len_x))])
        wins.append(row)

    return wins

def get_grid_overlap(image, numY, numX, overlap):
    '''
    
    This  Function receives one image and divides the 
    image into multiple windows based on numY and numX and Overlap. 
    Overlap is the amount of the windows that will be overlapped.  
    In this function calculates the widow location 
    depending on the overlap
    This returns a 2D array of smaller images(windows) that are overlapped
    '''
    
    height, width = image.shape
    len_y = int(height / numY)
    len_x = int(width / numX)
    
    overlapped_y = int(len_y * overlap / 100)
    overlapped_x = int(len_x * overlap / 100)
    
    if ((not(len_y * numY == height)) or (not(len_x * numX == width))):
        print("The no of window does not match with ith image size. Trimming the image and continue")
    
    wins = []
    i = 0
    j = 0
    
    while i < height:
        row = []
        j = 0
        
        while j < width:
            row.append(image[i: (i + len_y), j: (j + len_x)])
            j = j + overlapped_x
            
        wins.append(row)
        i = i + overlapped_y
        
    return wins

    ''' Perhaps better form...
    for i in range(0, height, 5):
        row = []
        for j in range(0, X, overlapped_x):
            row.append(image[i: (i + len_y), j: (j + len_x)])
        wins.append(row)

    return wins


    '''

def get_SearchArea(image, loc, winSize, winNo):
    '''
    
    This  Function Receives one image and returns a
    subsection of the image cented at loc and of 
    size depending on winsize and winNo. Where winsize
    is the size of a specified image window
    winNo is the number of windows for the search area
    dimension to be given by. eg. winNo= > search area 
    dimension is 3x3 image windows in size.
    This returns larger or smaller images(search area windows)
    '''
    height, width = image.shape

    # Ylen = int(winSize[0] * 2.5)
    # Xlen = int(winSize[1] * 2.5)

    len_y = int(winSize[0] * winNo / 2)
    len_x = int(winSize[1] * winNo / 2)

    min_y = loc[0] - len_y
    max_y = loc[0] + len_y
    min_x = loc[1] - len_x
    max_x = loc[1] + len_x

    if min_y < 0:
        min_y = 0
        return np.zeros( (len_y * 2, len_x * 2))
    if min_x < 0:
        min_x = 0
        return np.zeros( (len_y * 2, len_x * 2))

    if max_y >= height:
        max_y = height
        return np.zeros( (len_y * 2, len_x * 2))

    if max_x >= width:
        max_x = width
        return np.zeros( (len_y * 2, len_x * 2))

    return image[min_y:max_y, min_x:max_x]

def get_SearchArea_Horizontal(image, loc, winSize, winNo):
    '''

    This  Function Receives one image and returns 
    a subsection of the image cented at loc and 
    of size depending on winsize and winNo
    This is the same a gridSearchArea except only
    in the horizontal axis
    This returns larger or smaller images(search area windows)
    '''
    
    height, width = image.shape

    len_y = int(winSize[0] / 2)
    len_x = int(winSize[1] * winNo / 2)

    min_y = loc[0] - len_y
    max_y = loc[0] + len_y
    min_x = loc[1] - len_x
    max_x = loc[1] + len_x

    if min_y < 0:
        min_y = 0
        return np.zeros( (len_y * 2, len_x * 2))
    if min_x < 0:
        min_x = 0
        return np.zeros( (len_y * 2, len_x * 2))

    if max_y >= height:
        max_y = height
        return np.zeros( (len_y * 2, len_x * 2))

    if max_x >= width:
        max_x = width
        return np.zeros( (len_y * 2, len_x * 2))

    return image[min_y: max_y, min_x: max_x]

def get_CrossCorrelation(pattern, template, winSize, ygrid, xgrid):
    '''
    Task 5.
    This  Function Receives two images, pattern and 
    template. It searches for the pattern in the
    template using cross correlation and finds 
    the best match by finding maximum of the 
    correlation result
    This Function returns two numbers for y and 
    x axis indicating the respective distance  
    '''
    
    time_start = time.time()
    WinCenOrg = [ygrid, xgrid]
    nny, nnx = template.shape
    
    #import pdb; pdb.set_trace()
    print('getCrossCorrelation')
    #print(f'p = {pattern} t = {template}, w = {winSize} y = {ygrid} x = {xgrid}')
    
    corr = crr_2d(pattern, template)
        
    # corr = signal.correlate2d(template, pattern)
    # #corr = signal.correlate2d(template, pattern, boundary='symm', mode='same')
    # print(type(corr))

    dpy, dpx = np.unravel_index(np.argmax(corr), corr.shape)
    dpy = dpy - (nny / 2)
    dpx = dpx - (nnx / 2)

    time_end = time.time() - time_start
    print(time_end)

    print(dpy, dpx)
    return dpy, dpx  

def get_MultipassCrossCorrelation(pattern, template, winSize, ygrid, xgrid):
    '''
    
    This  Function Receives two images; pattern and 
    template. First searches for the pattern in the template 
    and gets distance between, then divides the pattern 
    into four parts and also reduces the 
    template size and searches for the individual 
    four sub sections and gets more accurate distance values.
    This function returns two (2x2) array for y 
    and x axis containing the respective distances  
    '''
    
    y, x = get_CrossCorrelation(pattern, template, winSize, ygrid, xgrid)
    pDim = pattern.shape
    SubPattern = get_grid(pattern, 2, 2)

    spDim = SubPattern[0][0].shape
    SearchWin = get_SearchArea(template, [ygrid + y, xgrid + x], spDim, 3)
    
    datax = np.zeros((4, 4))
    datay = np.zeros((4, 4))

    sig = [-1, 1]
    
    for i in range(0, 2):
        ddx = []
        ddy = []
        for j in range(0, 2):
            yy, xx = get_CrossCorrelation(SubPattern[i][j], template, spDim, \
                                          ygrid + y + (sig[j] * spDim[0]), \
                                          xgrid + x + (sig[j] * spDim[1]))
                                          
            datax[i][j] = xx
            datay[i][j] = yy

    return([datay, datax])

def ProcessImages(filepath):
    '''
   
    This  Function Receives the folderpath containing 
    two images and calculates the distance map using  
    get_MultipassCrossCorrelation
    This returns the distance map as a 2D array 
    '''
    
    pattern, template = load_images(filepath)
    
    if not(pattern.any() and template.any()):
        return False
        
    # no of rows an d column    
    nn = 100  

    wins = get_grid(pattern, nn, nn)
    leny, lenx = wins[0][0].shape
    dpy = np.zeros((leny * 2, lenx * 2))
    dpx = np.zeros((leny * 2, lenx * 2))

    distmap = []
    len_list = [ leny, lenx ]
    len_x_div_2 = lenx/2
    len_y_div_2 = leny/2

    for i in range(0, nn):
        TempWin_i = wins[i]
        tempWinCenX = int( (i*leny) + len_y_div_2 )
        TempWinCen = [ tempWinCenX, 0 ]

        for j in range(0, nn):
            TempWin = TempWin_i[j]            
            TempWinCen[1] = int( (j*lenx) + len_x_div_2 ) 

            #SearchWin = get_SearchArea_Horizontal( template, TempWinCen, len_list, 3)
            SearchWin = get_SearchArea(template, TempWinCen, len_list, 3 )

            if not len(SearchWin) > 1:
                continue
            y, x = get_MultipassCrossCorrelation(TempWin, SearchWin, len_list, \
                                                 TempWinCen[0], TempWinCen[1])
            print(y)

            for k in range(0, 2):
                for l in range(0, 2):
                    dpy[(i*4)+k][(j*4)+l] = abs(y[k][l])
                    dpx[(i*4)+k][(j*4)+l] = abs(x[k][l])

    print(dpy, dpx)

    return[dpy, dpx]


def ProcessImages2(filepath):
    '''
    
    This  Function Receives the folderpath containing 
    two image and calculates the distance map using  get_CrossCorrelation
    This returns the distance map as a 2D array 
    '''
    
    #import pdb; pdb.set_trace()   #debugging for reading in
    pattern, template = load_images(filepath)
    
    if not(pattern.any() and template.any()):
        return False
        
    # no of rows an d column
    nn = 30

    #import pdb; pdb.set_trace()
    wins = get_grid(pattern, nn, nn)
    leny, lenx = wins[0][0].shape
    dpy = np.zeros((leny, lenx))
    dpx = np.zeros((leny, lenx))

    # print(f' size of length {len(wins)} ')
    # print(f' length of first wins elements {lenx,leny} ')
    # print(f' size of change dpx dpy {dpx,dpy} ')


    distmap = []
    len_list = [ leny, lenx ]

    len_x_div_2 = lenx / 2
    len_y_div_2 = leny / 2

    for i in range(0, nn):
        TempWin_i = wins[i]
        #print(f' i = {i}')
        tempWinCenX = int( (i * leny) + len_y_div_2 )
        TempWinCen = [tempWinCenX, 0]
    
        for j in range(0, nn):
            # import pdb; 
            # if (j == 1 and i == 1):
            #     pdb.set_trace() 
            #print(f' j = {j}')
            
            TempWin = TempWin_i[j]            
            TempWinCen[1] = int( (j * lenx) + len_x_div_2 ) 

            #SearchWin = get_SearchArea_Horizontal( template, TempWinCen, len_list, 3)
            SearchWin = get_SearchArea(template, TempWinCen, len_list, 3)
            #call difeerent strategies here
            if not len(SearchWin) > 1:
                continue
            import pdb; pdb.set_trace() 
            y, x = get_CrossCorrelation( TempWin, SearchWin, len_list, \
                                         TempWinCen[0], TempWinCen[1] )
            dpy[i][j] = abs(y)
            dpx[i][j] = abs(x)
            #import pdb; pdb.set_trace() 

    print("\ndpy,dpx\n")
    print(dpy)
    print(dpx)
    return[dpy, dpx]

def main():
    '''
    Main implentation with toggle for single pass
    or multi pass correlation 
    The functions ProcessImages or ProcessImages2 can 
    be called with the file path as argument and receive 
    the distance map for plotting. 
    '''
    
    # switch using multi pass correlation or single
    multi = False  

    if multi:
        dpy, dpx = ProcessImages('..\\')   
    else:
        dpy, dpx = ProcessImages2('.\\')   
        
    plt.imshow(dpy, cmap='gray')
    plt.show()
    
    plt.imshow(dpx, cmap='gray')
    plt.show()

    x = dpx.shape[0]
    y = dpx.shape[1]
    z = np.zeros((x,y))
    for i in range(0, x):
        for j in range( 0, y):
            z[i][j] = np.sqrt((dpx[i][j])**2 + (dpy[i][j])*2)

    plt.imshow(z)    
    plt.show()

    # p = cv2.imread('left_box.tiff', cv2.IMREAD_COLOR)
    # t = cv2.imread('right_box.tiff', cv2.IMREAD_COLOR)
    # # a ,b, c = find_matches(t, p, thresh=None)
    # # print(a,b,c)
    # # plt.imshow(c)
    # # plt.show()
    # l_img = cv2.imread('left_box.tiff')
    # r_img = cv2.imread('right_box.tiff')    
    # l_XY = findClusterCenters(coordDF = getCoordDF(ori = 'right', dist = None), display = False)
    # r_XY = findClusterCenters(coordDF = getCoordDF(ori = 'right', dist = None), display = False)
    # print(l_XY)
    # print(r_XY)


if __name__ == '__main__':
    main()
