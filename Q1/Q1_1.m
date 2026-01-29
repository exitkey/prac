% (a). Degraded image
image = imread("C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q1\Degraded image.jpg");
img = im2gray(image);

% (b). Laplacian gradient image
sigma = 0.4;
alpha = 0.5;
laplacian = locallapfilt(img, sigma, alpha);

% (c). Laplacian masking
laplacian_mask = img + laplacian;

% (d). sobel gradient
sobel = edge(img, "sobel");

% (e). smoothing(sobel gradient image)
smoothing = smoothdata(img);

% (f). (c).*(e)
f = laplacian_mask * smoothing;

% (g). Synthesized image
Synthesized = img + sobel;

% (h). power-law(gamma) transformation
gamma = imadjust(img,[],[],0.5);

figure
subplot(2, 4, 1), imshow(img), title('(a). Degraded image')
subplot(2, 4, 2), imshow(laplacian), title('(b). Laplacian gradient image')
subplot(2, 4, 3), imshow(laplacian_mask), title('(c). Laplacian masking')
subplot(2, 4, 4), imshow(sobel), title('(d). sobel gradient')
subplot(2, 4, 5), imshow(smoothing), title('(e). smoothing(sobel gradient image)')
subplot(2, 4, 6), imshow(f), title('(f). (c).*(e)')
subplot(2, 4, 7), imshow(Synthesized), title('(g). Synthesized image')
subplot(2, 4, 8), imshow(gamma), title('(h). power-law(gamma) transformation')