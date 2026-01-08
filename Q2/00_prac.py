import numpy as np
from skimage.transform import rotate ## Image rotation routine.
import cv2
import scipy.fftpack as fft
import scipy.signal as sig
from PIL import Image


def build_laminogram(radonT):
    laminogram = np.zeros((radonT.shape[1],radonT.shape[1]))

    dTheta = 180.0 / radonT.shape[0]

    for i in range(radonT.shape[0]):
        temp = np.tile(radonT[i],(radonT.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram

def build_proj_ffts(projs):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.rfft(projs, axis=1)


def ramp_filter_ffts(ffts):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp


def build_proj_iffts(projs):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.irfft(projs, axis=1)


def hamming_window(projs):
    hamming = np.hamming(projs.shape[1])
    return hamming * projs

print("Loading...")

#PART 1
sinogram = cv2.imread(r"C:\Users\HEECHEOL\Desktop\grad\05_DL\02_DL\Q2\origin.jpg")
sinogram = cv2.cvtColor(sinogram, cv2.COLOR_BGR2GRAY)
sino_lam_part1 = build_laminogram(sinogram)
image1 = sino_lam_part1/sino_lam_part1.max()
print("Done calculating part 1")


#PART 2
sino_fft = build_proj_ffts(sinogram)
ramp_sino = ramp_filter_ffts(sino_fft)
sino_ifft_part2 = build_proj_iffts(ramp_sino)
sino_lam_part2 = build_laminogram(sino_ifft_part2)
image2 = sino_lam_part2/sino_lam_part2.max()
print("Done calculating part 2")


#PART 3
ham_sino = hamming_window(ramp_sino)
sino_ifft_part3 = build_proj_iffts(ham_sino)
sino_lam_part3 = build_laminogram(sino_ifft_part3)
image3 = sino_lam_part3/sino_lam_part3.max()
print("Done calculating part 3")


print("Press enter on image to progress through images")


cv2.imshow('Original Image',sinogram)
cv2.waitKey(0)

cv2.imshow("Part 1: Sinogram reconstruction from backprojections" ,image1)
cv2.waitKey(0)

cv2.imshow("Part 2: Sinogram reconstruction from backprojections" ,image2)
cv2.waitKey(0)

cv2.imshow("Part 3: Sinogram reconstruction from backprojections" ,image3)
cv2.waitKey(0)
