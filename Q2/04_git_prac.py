import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon

path = r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin_circle.png"

img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = img_u8.astype(np.float32) / 255.0

# 1) 빨간 테두리 제거 (두께에 맞게 6~12 사이 조절)
crop = 8
img = img[crop:-crop, crop:-crop]

# 2) 큰 원(phantom) 마스크: 흰 배경(1) 제외
mask0 = (img < 0.98).astype(np.uint8)
mask0 = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

num, labels, stats, _ = cv2.connectedComponentsWithStats(mask0, connectivity=8)
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
mask = (labels == largest)

# 3) 원 밖 0
img_masked = img.copy()
img_masked[~mask] = 0.0

# 4) (핵심) DC 제거 + 삽입물만 남기기
x = img_masked.copy()
mu = x[mask].mean()
x = x - mu
x[~mask] = 0.0
x = np.clip(-x, 0, None)     # 어두운 삽입물만 양수

# --- 검증(꼭 출력해봐) ---
print("outside mean/max:", x[~mask].mean(), x[~mask].max())
print("inside mean (after DC):", x[mask].mean())

# 5) radon
theta = np.linspace(0, 180, 180, endpoint=False)
sino = radon(x, theta=theta, circle=True)

# 6) plot
plt.figure(figsize=(6,5))
plt.imshow(sino.T, cmap="gray", aspect="auto", origin="lower")
plt.title("Sinogram (DC removed)")
plt.xlabel("r [index]")
plt.ylabel("theta [deg]")
plt.colorbar()
plt.tight_layout()
plt.show()