import numpy as np
import cv2
from skimage.transform import radon
import matplotlib.pyplot as plt

path = r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

h, w = img.shape
center_y, center_x = h // 2, w // 2
radius = min(h, w) // 2

y, x = np.ogrid[:h, :w]
mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
img[mask] = 0

sinogram = radon(img, circle=False)

sinogram_norm = cv2.normalize(sinogram, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


cv2.imshow("Sinogram", sinogram_norm)
cv2.waitKey(0)