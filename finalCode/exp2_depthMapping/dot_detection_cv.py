import cv2
import numpy as np


def plot_save(label):
    path = os.path.join("..","figures","calibration","fig_" + label + ".png")
    plt.savefig(path)


img = cv2.imread('test_images\\test_left_1.tiff')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

cnts = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

centers = []
if len(cnts) > 0:
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if x != None and y != None:
            centers.append((int(x), int(y)))

for center in centers:
    cv2.circle(img, center, 20, (0,255,0), 2)

print(centers)

cv2.imshow('image',img)
path = os.path.join("..","figures","calibration","fig_" + 'Dots_detected_OpenCV' + ".jpg")
cv2.imwrite('Dots_detected_OpenCV.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
