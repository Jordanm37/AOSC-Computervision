from crossCorrFunctions import *
import PIL

def main():
    orientations = ["left", "right"]
    # Provide a path to the left and right images 
    # eg. AOSC-Computervision-main/finalCode/exp2_depthMapping/uncalibImage
    uncalib = "uncalibImage/" 
   
    # Disparity of uncalibrated image
    min_disp = 16
    num_disp = 256-min_disp
    window_size = 8
    # stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)

    # Creating depth images fot the uncalibImage provided
    for tiffIm in ["box", "cone", "portal", "tuscany", "block"]:
    #for tiffIm in ["box", "cone", "portal", "tuscany"]:
        
        # process the path of image / read it / use grayscale 
        l_img = cv2.pyrUp(cv2.imread(uncalib+"left"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE))
        r_img = cv2.pyrUp(cv2.imread(uncalib+"right"+"_"+tiffIm+".tiff", cv2.IMREAD_GRAYSCALE))
        block_size = 16
        i = 8
        p1 = i * 3 * block_size
        p2 = i * 4 * 3 * block_size
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=10,
            speckleWindowSize=100,
            speckleRange=3,
            # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity = stereo.compute(l_img, r_img).astype(np.float32) / 16.0
        # import pdb; pdb.set_trace()
        disparity = ((disparity + 16)//5)
        message = f"{tiffIm}, quadric interpolation\n"\
            + f"p1: {p1}, p2: {p2}"
        print(message)

        # Create a new image - depth type
        img = PIL.Image.fromarray(disparity)
        plt.imshow(disparity, cmap='terrain', interpolation='quadric', resample=True)
        plt.title(message)
        plt.show()

        # for i in [ 'bilinear', 'bicubic', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell']:
        #     plt.imshow(disparity, cmap='terrain', interpolation=i)
        #     plt.title(f"Interpolation: {i}")
        #     plt.show()
        
        # for i in ['sinc', 'lanczos', 'blackman']:
        #     usable_rads =  [0.5 * r for r in range(2, 10)]
        #     for rad in usable_rads:
        #         plt.imshow(disparity, cmap='terrain', interpolation=i, filterrad=rad)
        #         plt.title(f"Interpolation: {i}, filterrad: {rad}")
        #         plt.show()

if __name__ == '__main__':
    main()