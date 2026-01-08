import cv2 as cv

# 1. 이미지 읽기 (흑백)
img = cv.imread(
    'C:/Users/HEECHEOL/Desktop/grad/05_DL/01_DL/Q1_251229/Degraded image.jpg',
    cv.IMREAD_GRAYSCALE
)
assert img is not None, "file could not be read, check with os.path.exists()"

# 2. 필터 적용
laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# 3. 보기 좋게 변환 (중요!)
laplacian = cv.convertScaleAbs(laplacian)
sobelx = cv.convertScaleAbs(sobelx)
sobely = cv.convertScaleAbs(sobely)

# 4. OpenCV 창으로 시각적 확인
cv.imshow("Original", img)
cv.imshow("Laplacian", laplacian)
cv.imshow("Sobel X", sobelx)
cv.imshow("Sobel Y", sobely)

# 5. 키 입력 대기 후 종료
cv.waitKey(0)
cv.destroyAllWindows()