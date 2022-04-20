import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('C:/Users/Berto/p3input.jpg', 0) # load an image

#Output is a 2D complex array. 1st channel real and 2nd imaginary
#For fft in opencv input image needs to be converted to float32
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

#Rearranges a Fourier transform X by shifting the zero-frequency 
#component to the center of the array.
#Otherwise it starts at the tope left corenr of the image (array)
dft_shift = np.fft.fftshift(dft)

##Magnitude of the function is 20.log(abs(f))
#For values that are 0 we may end up with indeterminate values for log. 
#So we can add 1 to the array to avoid seeing a warning. 
magnitude_spectrum = 2000 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])+0.0001)

"""
generating ideal low pass filter
"""
#M and N
M, N = img.shape

#center1 and center2 -> P and Q 
M_cen, N_cen = int(M / 2), int(N / 2)

#creating complex mask for ideal low pass filter
mask = np.zeros((M, N, 2), np.uint8)

#cutoff frequency
d_0 = 50

#creating center array
center = [M_cen, N_cen]

#returns an open multi-dimensional "meshgrid"
x, y = np.ogrid[:M, :N]

#ideal low pass filter
mask_area1 = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** (1/2) <= d_0
mask[mask_area1] = 1
mask_area2 = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** (1/2) > d_0
mask[mask_area2] = 0

#apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 0.0001)

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back_mag = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.title.set_text('FFT using libraries')
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(img_back_mag, cmap='gray')
ax3.title.set_text('IFFT using libraries, d_0 = 50')
plt.show()

