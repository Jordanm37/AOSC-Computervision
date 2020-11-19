# -*- coding: utf-8 -*-

# importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\Jordan Moshcovitis\\Documents\\Masters\\The Art of Scientific Computing\\AOSC-Computervision\\testingRetiredCode\\Part_3\\2.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
m_s = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(m_s)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
