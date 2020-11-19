from crossCorrFunctions import *
import PIL

def main():
    orientations = ["left", "right"]
    uncalib = "uncalibImage\\" 
    #Disparity of uncalibrated image
    min_disp = 32
    num_disp = 112-min_disp
    window_size = 5

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)

    for tiffIm in ["box","cone","portal","tuscany", "block"]:
        l_img = cv2.imread(uncalib+"left"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE)
        r_img = cv2.imread(uncalib+"right"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE)
        disparity = stereo.compute(l_img, r_img)
        disparity = ((disparity + 16)//5)
        img = PIL.Image.fromarray(disparity)
        plt.imshow(disparity, cmap='terrain')
        plt.title(tiffIm)
        plt.show()

    # uncalib = "cal_images\\" 
    # #Disparity of uncalibrated image
    # min_disp = 32
    # num_disp = 112-min_disp
    # window_size = 6

    # stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)

    # for tiffIm in ["2000", "1980", "1960", "1940", "1920", "1900"]:
    #     l_img = cv2.imread(uncalib+"left"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE)
    #     r_img = cv2.imread(uncalib+"right"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE)
    #     disparity = stereo.compute(l_img, r_img)
    #     disparity = ((disparity + 16)//5)
    #     img = PIL.Image.fromarray(disparity)
    #     plt.imshow(disparity, cmap='terrain')
    #     plt.title(tiffIm)
    #     plt.show()


if __name__ == '__main__':
    main()
