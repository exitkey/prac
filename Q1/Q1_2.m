% input image (경로는 작은따옴표 추천)
image = imread('C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q1\Degraded image.jpg');
img = im2double(im2gray(image)); % im2double: 출력값을 정수형 데이터에서 [0, 1] 범위로 다시 스케일링

% (b) Laplacian image
lap = imfilter(img, fspecial('laplacian'), 'replicate');


% (c) Laplacian masking
c = img + lap;

% (d) Sobel gradient magnitude
sobel = imgradient(img, 'sobel');

% (e) smoothing(sobel gradient image)
e = imgaussfilt(sobel, 1);

% (f) (c).*(e)
f = c .* e;

% (g) Synthesized
g = img + sobel;

% (h) power-law(gamma) transform
h = imadjust(g, [], [], 0.5);

figure
subplot(2,4,1), imshow(img), title('(a). Degraded image')
subplot(2,4,2), imshow(lap), title('(b). Laplacian image')
subplot(2,4,3), imshow(c), title('(c). Laplacian masking')
subplot(2,4,4), imshow(sobel), title('(d). Sobel gradient magnitude')
subplot(2,4,5), imshow(e), title('(e). Smoothed gradient')
subplot(2,4,6), imshow(f), title('(f). (c).*(e)')
subplot(2,4,7), imshow(g), title('(g). Synthesized image')
subplot(2,4,8), imshow(h), title('(h). power-law(Gamma) transform')
