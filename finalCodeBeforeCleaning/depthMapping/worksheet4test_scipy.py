import glob
from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

from test_xcorr import TemplateMatch



# Task 1
def load_images(filepath):
    '''
    Task 1.
    This  Function reads two images in the folder; one 
    with '_right' in filename another in '_left' in filename. 
    Convert the two images to numpy array and convert 
    from rgb to blackand white and return the two images 
    as pattern and template
    Return false if there is no file with 
    '_right' or '_left' in the given folder
    '''
    template_fn = ""
    pattern_fn = ""
    fileNames = glob.glob(filepath+"*.tiff*")
    for fileName in fileNames:
        print(fileName)
        if "right" in fileName:  # right_ is for the file type in the test folder may be removed if files are named properly
            template_fn = fileName
        elif "left" in fileName:
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
        template = 0.2989 * template[:, :, 0] + 0.5870 * \
            template[:, :, 0] + 0.1140 * template[:, :, 0]
        pattern = 0.2989 * pattern[:, :, 0] + 0.5870 * \
            pattern[:, :, 0] + 0.1140 * pattern[:, :, 0]

    except expression as identifier:
        print("Error in Converting from RGB to Gray scale")
        return False

    return pattern, template


# Task 2.a
def get_grid(image, numY, numX):
    '''
    Task 2.a.
    This  Function receives one image and divides the image 
    into multiple windows based on numY and numX as division 
    of widows in y and x axis 
    This returns a 2D array of smaller images(windows)
    '''
    #import pdb; pdb.set_trace()
    Y, X = image.shape
    Ylen = int(Y / numY) #extra size +1 drops off if present becasue of integer division
    Xlen = int(X / numX) # ""

    wins = []
    for i in range(0, numY):
        row = []
        for j in range(0, numX):
            row.append(image[(i * Ylen): (((i + 1) * Ylen)),
                            (j * Xlen): (((j + 1) * Xlen))])
        wins.append(row)

    return wins


# Task 2.a
def get_grid_overlap(image, numY, numX, overlap):
    '''
    Task 2.b.
    This  Function receives one image and divides the 
    image into multiple windows based on numY and numX and Overlap. 
    Overlap is the amount of the windows that will be overlapped.  
    In this function calculates the widow location 
    depending on the overlap
    This returns a 2D array of smaller images(windows) that are overlapped
    '''
    Y, X = image.shape
    Ylen = int(Y / numY)
    Xlen = int(X / numX)
    Yol = int(Ylen * overlap / 100)
    Xol = int(Xlen * overlap / 100)
    if ((not(Ylen * numY == Y)) or (not(Xlen * numX == X))):
        print("The no of window does not match with ith image size. Trimming the image and continue")
    wins = []
    i = 0
    j = 0
    while i < Y:
        row = []
        j = 0
        while j < X:
            row.append(image[i: (i + Ylen), j: (j + Xlen)])
            j = j + Xol
        wins.append(row)
        i = i + Yol
    return wins

    ''' Perhaps better form...
    for i in range(0, Y, 5):
        row = []
        for j in range(0, X, Xol):
            row.append(image[i: (i + Ylen), j: (j + Xlen)])
        wins.append(row)

    return wins


    '''



# Task 3.a


def get_SearchArea(image, loc, winSize, winNo):
    '''
    Task 3.a.
    This  Function Receives one image and returns a
    subsection of the image cented at loc and of 
    size depending on winsize and winNo. Where winsize
    is the size of a specified image window
    winNo is the number of windows for the search area
    dimension to be given by. eg. winNo= > search area 
    dimension is 3x3 image windows in size.
    This returns larger or smaller images(search area windows)
    '''
    Y, X = image.shape

    # Ylen = int(winSize[0] * 2.5)
    # Xlen = int(winSize[1] * 2.5)

    Ylen = int(winSize[0] * winNo / 2)
    Xlen = int(winSize[1] * winNo / 2)

    Ymin = loc[0] - Ylen
    Ymax = loc[0] + Ylen
    Xmin = loc[1] - Xlen
    Xmax = loc[1] + Xlen

    if Ymin < 0:
        Ymin = 0
        return np.zeros( (Ylen * 2, Xlen * 2))
    if Xmin < 0:
        Xmin = 0
        return np.zeros( (Ylen * 2, Xlen * 2))

    if Ymax >= Y:
        Ymax = Y
        return np.zeros( (Ylen * 2, Xlen * 2))

    if Xmax >= X:
        Xmax = X
        return np.zeros( (Ylen * 2, Xlen * 2))

    return image[Ymin:Ymax, Xmin:Xmax]


# Task 3.b
def get_SearchArea_Horizontal(image, loc, winSize, winNo):
    '''
    Task 3.a.
    This  Function Receives one image and returns 
    a subsection of the image cented at loc and 
    of size depending on winsize and winNo
    This is the same a gridSearchArea except only
    in the horizontal axis
    This returns larger or smaller images(search area windows)
    '''
    Y, X = image.shape

    Ylen = int(winSize[0] / 2)
    Xlen = int(winSize[1] * winNo / 2)

    Ymin = loc[0] - Ylen
    Ymax = loc[0] + Ylen
    Xmin = loc[1] - Xlen
    Xmax = loc[1] + Xlen

    if Ymin < 0:
        Ymin = 0
    return []
    if Xmin < 0:
        Xmin = 0
    return []

    if Ymax >= Y:
        Ymax = Y
    return []

    if Xmax >= X:
        Xmax = X
    return []

    return image[Ymin: Ymax, Xmin: Xmax]

# Task 5


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
    print(f'p = {pattern} t = {template}, w = {winSize} y = {ygrid} x = {xgrid}')
    corr = signal.correlate2d(template, pattern)
    #corr = signal.correlate2d(template, pattern, boundary='symm', mode='same')
    print(type(corr))

    dpy, dpx = np.unravel_index(np.argmax(corr), corr.shape)
    dpy = dpy - (nny / 2)
    dpx = dpx - (nnx / 2)

    time_end = time.time() - time_start
    print(time_end)

    print(dpy, dpx)
    return dpy, dpx  
[ x - - - -]
[ - x - - -]

# Task 4
def get_MultipassCrossCorrelation(pattern, template, winSize, ygrid, xgrid):
    '''
    Task 4.
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
            yy, xx = get_CrossCorrelation(SubPattern[i][j], template, spDim, ygrid + y + (
                sig[j] * spDim[0]), xgrid + x + (sig[j] * spDim[1]))
            datax[i][j] = xx
            datay[i][j] = yy

    return([datay, datax])


#Task 6a.
def ProcessImages(filepath):
    '''
    Task 6a.
    This  Function Receives the folderpath containing 
    two images and calculates the distance map using  
    get_MultipassCrossCorrelation
    This returns the distance map as a 2D array 
    '''
    pattern, template = load_images(filepath)
    if not(pattern.any() and template.any()):
        return False
    nn = 10  # no of rows an d column

    wins = get_grid(pattern, nn, nn)
    leny, lenx = wins[0][0].shape
    dpy = np.zeros((leny * 2, lenx * 2))
    dpx = np.zeros((leny * 2, lenx * 2))

    distmap = []

    for i in range(0, nn):
        #dd = []
        for j in range(0, nn):
            TempWin = wins[i][j]

            TempWinCen = [int((i * leny) + (leny / 2)),
                          int((j * lenx) + (lenx / 2))]
            SearchWin = get_SearchArea_Horizontal(
                template, TempWinCen, [leny, lenx], 2)

            if not len(SearchWin) > 1:
                continue
            y, x = generate_MultipassCrossCorrelation(
                TempWin, SearchWin, [leny, lenx], TempWinCen[0], TempWinCen[1])
            print(y)

            for k in range(0, 2):
                for l in range(0, 2):
                    dpy[(i*4)+k][(j*4)+l] = abs(y[k][l])
                    dpx[(i*4)+k][(j*4)+l] = abs(x[k][l])

    print(dpy, dpx)

    return[dpy, dpx]


#task 6.b.
def ProcessImages2(filepath):
    '''
    Task 6b.
    This  Function Receives the folderpath containing 
    two image and calculates the distance map using  get_CrossCorrelation
    This returns the distance map as a 2D array 
    '''
    #import pdb; pdb.set_trace()   #debugging for reading in
    pattern, template = load_images(filepath)
    if not(pattern.any() and template.any()):
        return False
    nn = 10  # no of rows an d column

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

    len_x_div_2 = lenx/2
    len_y_div_2 = leny/2

    for i in range(0, nn):
        dd = []
        TempWin_i = wins[i]
        #print(f' i = {i}')
        tempWinCenX = int( (i*leny) + len_y_div_2 )
        TempWinCen = [ tempWinCenX, 0 ]
    
        for j in range(0, nn):
            # import pdb; 
            # if (j == 1 and i == 1):
            #     pdb.set_trace() 
            print(f' j = {j}')
            TempWin = TempWin_i[j]
      
            TempWinCen[1] = int( (j*lenx) + len_x_div_2 ) 
            
            SearchWin = get_SearchArea(template, TempWinCen, len_list, 3 )
            
            if not len(SearchWin) > 1:
                continue

            y, x = get_CrossCorrelation( TempWin, SearchWin, len_list, TempWinCen[0], TempWinCen[1] )
            dpy[i][j] = abs(y)
            dpx[i][j] = abs(x)
            #import pdb; pdb.set_trace() 
    print("\ndpy,dpx\n")
    print(dpy, dpx)
    return[dpy, dpx]


#Unoptimsed old block
    # for i in range(0, nn):
    #     dd = []
    #     TempWin_i = wins[i]
    #     #print(f' i = {i}')
    #     for j in range(0, nn):
    #         # import pdb; 
    #         # if (j == 1 and i == 1):
    #         #     pdb.set_trace() 
    #         #print(f' j = {j}')
    #         TempWin = TempWin_i[j]

    #         TempWinCen = [int((i*leny)+(leny/2)), int((j*lenx)+(lenx/2))]
    #         SearchWin = get_SearchArea(template, TempWinCen, [leny, lenx], nn)
    #         if not len(SearchWin) > 1:
    #             continue
    #         y, x = get_CrossCorrelation(
    #             TempWin, SearchWin, [leny, lenx], TempWinCen[0], TempWinCen[1])
    #         dpy[i][j] = abs(y)
    #         dpx[i][j] = abs(x)
    #         #import pdb; pdb.set_trace() 

    # print(dpy, dpx)
    # return[dpy, dpx]


def main():
    '''
    Main implentation with toggle for single pass
    or multi pass correlation 
    The functions ProcessImages or ProcessImages2 can 
    be called with the file path as argument and receive 
    the distance map for plotting. 
    '''
    multi = False  # switch using multi pass correlation or single

    if multi:
        dpy, dpx = ProcessImages(
            #'C:\\Users\\Jordan\\Documents\\GitHub\\AOSC-Computervision\\2_depth_mapping\\')
            '..\\2_depth_mapping\\')   
    else:
        dpy, dpx = ProcessImages2(
        #    'C:\\Users\\Jordan\\Documents\\GitHub\\AOSC-Computervision\\Image_testing\\')
              '.\\')            
    plt.imshow(dpy, cmap='gray')
    plt.show()


if __name__ == '__main__':

    main()

