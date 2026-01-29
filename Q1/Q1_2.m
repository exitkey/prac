% (a). Degraded image
image = imread('C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q1\Degraded image.jpg');
img = im2double(im2gray(image)); % im2double: 출력값을 정수형 데이터에서 [0, 1] 범위로 다시 스케일링

% (b). Laplacian image
laplacian = imfilter(img, fspecial('laplacian'), 'replicate');

% (c). Laplacian masking
laplacian_mask = img + laplacian;

% (d). Sobel gradient
sobel = imgradient(img, 'sobel');

% (e). smoothing(sobel gradient image)
smoothing = imgaussfilt(sobel, 1);

% (f). (c).*(e)
f = laplacian_mask .* smoothing;

% (g). Synthesized image
Synthesized = img + sobel;

% (h). power-law(gamma) transform
gamma = imadjust(Synthesized, [], [], 0.5);

figure
subplot(2,4,1), imshow(img), title('(a). Degraded image')
subplot(2,4,2), imshow(laplacian), title('(b). Laplacian image')
subplot(2,4,3), imshow(laplacian_mask), title('(c). Laplacian masking')
subplot(2,4,4), imshow(sobel), title('(d). Sobel gradient magnitude')
subplot(2,4,5), imshow(smoothing), title('(e). Smoothed gradient')
subplot(2,4,6), imshow(f), title('(f). (c).*(e)')
subplot(2,4,7), imshow(Synthesized), title('(g). Synthesized image')
subplot(2,4,8), imshow(gamma), title('(h). power-law(Gamma) transform')
