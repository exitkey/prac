import numpy as np
import cv2
from matplotlib import pyplot as plt
 
def convert_gray255(image):
    image_r = 255*(image - image.min())/(image.max() - image.min())
    return(image_r)
 
img = cv2.imread("Degraded image.jpg", 0)
laplacian = convert_gray255(cv2.Laplacian(img, cv2.CV_64F))
img_laplacian = convert_gray255(img + laplacian)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel=np.clip(np.abs(sobelx) + np.abs(sobely),0,255)
blur = cv2.bilateralFilter(img,9,75,75)
masking = convert_gray255(blur * img_laplacian)
img_masking =convert_gray255(img + masking)
img_result = 1*((img_masking)**0.5)
 
plt.subplot(2, 4, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 3), plt.imshow(img_laplacian, cmap='gray')
plt.title('img_laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 4), plt.imshow(sobel, cmap='gray')
plt.title('sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 5), plt.imshow(blur, cmap='gray')
plt.title('blur'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 6), plt.imshow(masking, cmap='gray')
plt.title('masking'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 7), plt.imshow(img_masking, cmap='gray')
plt.title('img_masking'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 4, 8), plt.imshow(img_result, cmap='gray')
plt.title('img_result'), plt.xticks([]), plt.yticks([])
 
plt.show()