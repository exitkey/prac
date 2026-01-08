from skimage.transform import radon
import cv2
import numpy as np



img = cv2.imread(r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin.jpg", 0)  # read image in grayscale


I = img - np.mean(img)
sinogram = radon(I)

cv2.imshow("sinogram", sinogram)