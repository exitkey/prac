import cv2 # import: 불러오기
import numpy as np # 'numpy'를 'np'로 호칭
from matplotlib import pyplot as plt # 'matplotlib'이라는 라이브러리에서 'pyplot'이라는 모듈만 사용
import textwrap

# (a) Degraded image
img = cv2.imread('C:/Users/HEECHEOL/Desktop/grad/05_DL/01_DL/Q1_251229/Degraded image.jpg', flags=0)

# (b) Laplacian gradient image
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# (c) Laplacian masking
AddLap = img + laplacian
imgAddLap = np.uint8(
    cv2.normalize(AddLap, None, 0, 255, cv2.NORM_MINMAX)
)

# (d) Sobel gradient
SobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)
SobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(SobelX)
absY = cv2.convertScaleAbs(SobelY)

SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
imgSobel = np.uint8(
    cv2.normalize(SobelXY, None, 0, 255, cv2.NORM_MINMAX)
)
# CV_16S: 16비트의 signed 함수(부호가 있는 함수, 미분 결과는 음수도 나올 수 있기 때문)
# convertScaleAbs: 절대값 계산, 타입 변환

# (e) Smoothing(sobel gradient image)
kernelBox = np.ones((5, 5), np.float32) / (5 * 5)
SobelBox = cv2.filter2D(imgSobel, -1, kernelBox)
imgSobelBox = np.uint8(
    cv2.normalize(SobelBox, None, 0, 255, cv2.NORM_MINMAX)
)

# (f) (c).*(e)
mask = AddLap * SobelBox
imgMask = np.uint8(
    cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
)

# (g) Synthesized image
passivation = img + imgSobel
imgPassi = np.uint8(
    cv2.normalize(passivation, None, 0, 255, cv2.NORM_MINMAX)
)

# (h) power-law(gamma) transformation
epsilon = 1e-5
Gamma = np.power(imgPassi + epsilon, 0.5)
imgGamma = np.uint8(
    cv2.normalize(Gamma, None, 0, 255, cv2.NORM_MINMAX)
)
# 어두워서 안 보이는 부분을 밝게 끌어 올려주는 역할
# epsilon: 컴퓨터 공학에서 무시할 수 있을만큼 작지만 0은 아닌 양수를 뜻하는 변수명
# np.power: 거듭제곱 계산 함수
# 지수 = 감마값 = 0.5


plt.figure(figsize=(10, 7))
# 출력창 크기를 inch 단위로 표현

titleList = [
    "(a) Degraded image",
    "(b) Laplacian gradient image",
    "(c) Laplacian masking",
    "(d) Sobel gradient",
    "(e) Smoothing(sobel gradient image)",
    "(f) (c).*(e)",
    "(g) Synthesized image",
    "(h) power-law(gamma) transformation"
]

imageList = [
    img,
    laplacian,
    imgAddLap,
    imgSobel,
    imgSobelBox,
    imgMask,
    imgPassi,
    imgGamma
]

for i in range(8): # i는 0부터 7까지
    plt.subplot(2, 4, i + 1) #2행 4열, i+1 번째에 배치
    wrapped_title = "\n".join(textwrap.wrap(titleList[i], width=20))
    plt.title(wrapped_title, y=-0.23, fontsize=10)
    plt.axis('off') # 눈금, 좌표축 제거
    plt.imshow(imageList[i], cmap='gray')

plt.subplots_adjust(wspace=0.3, hspace=0.4, bottom=0.1)
# plt.tight_layout() # 자동 레이아웃 정리
plt.show() # 화면에 표시