from crossCorrFunctions import *
import PIL

def main():

    uncalib = "cal_images\\cal_image_" 
    #Disparity of uncalibrated image
    min_disp = 16
    num_disp = 256-min_disp
    window_size = 8

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)

    for tiffIm in ["2000", "1980", "1960", "1940", "1920", "1900"]:
        l_img = cv2.imread(uncalib+"left"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE)
        r_img = cv2.imread(uncalib+"right"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE)
        disparity = stereo.compute(l_img, r_img)
        disparity = ((disparity + 16)//5)
        img = PIL.Image.fromarray(disparity)
        plt.imshow(disparity, cmap='gray')
        plt.title(tiffIm)
        plt.show()

if __name__ == '__main__':
    main()
