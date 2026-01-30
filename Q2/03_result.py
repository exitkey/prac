import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon
from skimage.util import invert


# Load image
img_u8 = cv2.imread(r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin_circle.png", cv2.IMREAD_GRAYSCALE)
img = img_u8.astype(np.float32) # astype : 자료형 변경
img_inv = invert(img)  # Invert image for radon transform

# Generate sinogram
theta = np.linspace(0.0, 180.0, 180, endpoint=False)
sino = radon(img_inv, theta=theta, circle=False)
sino_plot = sino.T # Transpose for correct orientation

# Define extents in mm
h, w = img.shape # img.shape : (세로 길이, 가로 길이)
pixel_size_mm=1.0
x_extent_mm = (-(w / 2) * pixel_size_mm, (w / 2) * pixel_size_mm) # x축 범위 설정
y_extent_mm = (-(h / 2) * pixel_size_mm, (h / 2) * pixel_size_mm) # y축 범위 설정

# Sinogram extent
num_r = sino.shape[0] # 사이노그램의 r 방향 샘플 수, 세로 길이(shape[행 개수, 열 개수])
r_max_mm = (num_r / 2) * pixel_size_mm
r_extent_mm = (-r_max_mm, r_max_mm)

# Imaging
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

ax1.set_title("Disk Phantoms")
ax1.set_xlabel("x [mm]")
ax1.set_ylabel("y [mm]")
ax1.imshow(
    img,
    cmap="gray",
    extent=(x_extent_mm[0], x_extent_mm[1], y_extent_mm[0], y_extent_mm[1]),
    )

ax2.set_title("Sinogram")
ax2.set_xlabel("r [mm]")
ax2.set_ylabel("[degree]")
ax2.imshow(
    sino_plot,
    cmap="gray",
    extent=(r_extent_mm[0], r_extent_mm[1], theta[0], theta[-1]), # theta[-1] : theta의 마지막 값
    origin="lower",
    aspect="auto",
    )

# Adjust layout and show plot
plt.tight_layout()
plt.show()