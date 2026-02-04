import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon

# Load image
img_u8 = cv2.imread(r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin.jpg", cv2.IMREAD_GRAYSCALE)
img = img_u8.astype(np.float32) # astype : 자료형 변경
img = img / 255.0  # Normalize to [0, 1]
img_inv = 1 - img  # Invert image

# Generate sinogram
theta = np.linspace(0.0, 180.0, 180, endpoint=False)
sino = radon(img_inv, theta=theta, circle=False)
sino_plot = sino.T # 가로축, 세로축 변경

# Define extents in mm
h, w = img.shape # img.shape : (세로 길이, 가로 길이)
pixel_size_mm = 1.0 # 1 픽셀은 1 mm라고 가정
x_extent_mm = (-(w / 2) * pixel_size_mm, (w / 2) * pixel_size_mm) # x축 범위 설정
y_extent_mm = (-(h / 2) * pixel_size_mm, (h / 2) * pixel_size_mm) # y축 범위 설정

# Sinogram extent
num_r = sino.shape[0] # 사이노그램의 r 방향 샘플 수, 세로 길이(shape[행 개수, 열 개수])
r_max_mm = (num_r / 2) * pixel_size_mm # r축(mm): 대략 중심 기준으로 [-R, +R]로 놓고 보기 좋게 맞춤
r_extent_mm = (-r_max_mm, r_max_mm) #sinogram의 r축을 검출기 중심 기준의 실제 거리 좌표(mm)로 해석하기 위한 시각화 설정

# Imaging
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

ax1.set_title("Disk Phantoms")
ax1.set_xlabel("x [mm]")
ax1.set_ylabel("y [mm]")
ax1.imshow(
    img,
    cmap="gray",
    extent=(x_extent_mm[0], x_extent_mm[1], y_extent_mm[0], y_extent_mm[1]), # (x축 최소, x축 최대, y축 최소, y축 최대)
    )

ax2.set_title("Sinogram")
ax2.set_xlabel("r [mm]")
ax2.set_ylabel("[degree]")
ax2.imshow(
    sino_plot,
    cmap="gray_r", # 역색상 맵 사용
    extent=(r_extent_mm[0], r_extent_mm[1], theta[0], theta[-1]), # theta[-1] : theta의 마지막 값
    origin="lower",
    aspect="auto", # 축 비율 자동, 안 쓰면 sinogram이 찌그러져 보일 수 있음
    )

# Adjust layout and show plot
plt.tight_layout() # 제목/축라벨이 겹치지 않게 자동 배치
plt.show()