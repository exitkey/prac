print("==============================")
print("1. 프로그램 시작! (이게 보이면 파이썬은 정상)")
print("==============================")

import numpy as np
import cv2
from skimage.transform import radon
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin.jpg", flags = 0)

if img is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

h, w = img.shape
center_y, center_x = h // 2, w // 2
radius = min(h, w) // 2

y, x = np.ogrid[:h, :w]
mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
img[mask] = 0

theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
sinogram = radon(img, circle=False, theta=theta)
sinogram_norm = cv2.normalize(sinogram, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))

ax1.set_title("Disk Phantoms")
ax1.imshow(img, cmap = plt.cm.Greys_r)

ax2.set_title("Sinogram")
ax2.set_xlabel("r[mm]")
ax2.set_ylabel("[degree]")
ax2.imshow(sinogram_norm)

fig.tight_layout()
plt.show()