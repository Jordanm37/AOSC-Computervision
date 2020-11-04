import glob
from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# Task 1
def load_images(filepath):
    '''
    Task 1.
    This  Function reads two images in the folder one with right in filename another in left in filename. 
    Convert the two images to numpy array and convert from rgb to blackand white 
    and Return the two imagages as pattern and template
    This will return false if there is no file with right and left in the given folder
    '''
    template_fn = ""
    pattern_fn = ""
    fileNames = glob.glob(filepath+"*.*")
    for fileName in fileNames:
        print(fileName)
        if "right" in fileName:  # right_ is for the file type in the test folder may beremoved if files are named properly
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
        print("Error in Convertin from RGB to Gray scale")
        return False

    return pattern, template


# Task 2.a
def get_grid(image, numY, numX):
    '''
    Task 2.a.
    This  Function Receives one image and devide the image to multiple windows based on numY and numX as number of widows in y and x  axis 
    This reurns a 2D array of smaller images(windows)
    '''
    Y, X = image.shape
    Ylen = int(Y / numY)
    Xlen = int(X / numX)

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
    This  Function Receives one image and devide the image to multiple windows based on numY and numX  and Overlap
    In this function calculates the widow location depending on the overlap
    This reurns a 2D array of smaller images(windows)
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

# Task 3.a


def get_SearchArea(image, loc, winSize, winNo):
    '''
    Task 3.a.
    This  Function Receives one image and retuns an subsection of the image cented at loc and of size depencing on winsize and winNo
    winSize is the size of the original windo. and win no is the no of repetation  
    This reurns  smaller images(search area windows)
    '''
    Y, X = image.shape

    Ylen = int(winSize[0] * winNo / 2)
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

    return image[Ymin:Ymax, Xmin:Xmax]


# Task 3.b
def get_SearchArea_Horizontal(image, loc, winSize, winNo):
    '''
    Task 3.a.
    This  Function Receives one image and retuns an subsection of the image cented at loc and of size depencing on winsize and winNo
    winSize is the size of the original windo. and win no is the no of repetation . But in this case repetation is considered only for horizontal axis 
    This reurns  smaller images(search area windows)
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
    This  Function Receives two image pattern and template search the pattern in template using cross corrilation and find the best match by finding maximam of the corrilation result
    This Function returns two numbers for y and x axis indicating the respecive distance  
    '''
    WinCenOrg = [ygrid, xgrid]
    nny, nnx = template.shape

    corr = signal.correlate2d(template, pattern, boundary='symm', mode='same')
    dpy, dpx = np.unravel_index(np.argmax(corr), corr.shape)
    dpy = dpy - (nny / 2)
    dpx = dpx - (nnx / 2)
    print(dpy, dpx)
    return [dpy, dpx]


# Task 4
def get_MultipassCrossCorrelation(pattern, template, winSize, ygrid, xgrid):
    '''
    Task 4.
    This  Function Receives two image pattern and template first search the pattern in template and get an distantance then devide the pattern in 4 parts and also reduce the 
    template size and searches for the individual 4 sub section and get more accurate distance values.
    this Function returns two (2x2) array for y and x axis contingin the respecive distance  
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
    This  Function Receives the folderpath containing two image and calulates the distance map using  get_MultipassCrossCorrelation
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
        dd = []
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
    This  Function Receives the folderpath containing two image and calulates the distance map using  get_CrossCorrelation
    This returns the distance map as a 2D array 
    '''
    pattern, template = load_images(filepath)
    if not(pattern.any() and template.any()):
        return False
    nn = 10  # no of rows an d column

    wins = get_grid(pattern, nn, nn)
    leny, lenx = wins[0][0].shape
    dpy = np.zeros((leny, lenx))
    dpx = np.zeros((leny, lenx))

    distmap = []

    for i in range(0, nn):
        dd = []
        for j in range(0, nn):
            TempWin = wins[i][j]

            TempWinCen = [int((i*leny)+(leny/2)), int((j*lenx)+(lenx/2))]
            SearchWin = get_SearchArea(template, TempWinCen, [leny, lenx], 2)
            if not len(SearchWin) > 1:
                continue
            y, x = get_CrossCorrelation(
                TempWin, SearchWin, [leny, lenx], TempWinCen[0], TempWinCen[1])
            dpy[i][j] = abs(y)
            dpx[i][j] = abs(x)

    print(dpy, dpx)
    return[dpy, dpx]


def main():
    '''
    Final Implimentation 
    The functions ProcessImages or ProcessImages2 can be called with the file path as ardumant and receive the distance map for plotting or other uses 
    '''
    multi = True  # switch using multi pass correlation or single

    if multi:
        dpy, dpx = ProcessImages(
            'C:\\Users\\Jordan\Desktop\\AOSC\\Part2\\data1\\')
    else:
        dpy, dpx = ProcessImages2(
            'C:\\Users\\Jordan\Desktop\\AOSC\\Part2\\data1\\')

    plt.imshow(dpy, cmap='gray')
    plt.show()


if __name__ == '__main__':

    main()
