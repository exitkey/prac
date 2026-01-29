I = imread('C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin_circle.png');
img = im2double(im2gray(I));

theta = 0:180;
[R,xp] = radon(img,theta);

figure
R_T = R';
imagesc(xp, theta, R_T)
title("Sinogram")
axis xy
colormap(gray)
colorbar
xlabel('r[mm]')
ylabel('\theta[degree]')
set(gca, "XTick", -200:100:200)
set(gca, "YTick", 0:20:160)