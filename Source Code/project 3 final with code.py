import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
Six total steps to complete this project

1.) input image: f(x,y) is of size M*N. Zero-pad it to fp(x,y). The new image size is P*Q, where P = 2*M - 1 and Q = 2*N - 1
2.) perform 2D FFT of the zero padded array
3.) shift the origin from the top left of the array to the absolute middle of the array
4.) apply an ideal low pass filter to the shifted array
5.) perform 2D IFFT to the new array produced in the previous step
6.) extract the M*N sub-region from the top-left quadrant of the 2D IFFT array
"""

#function to zero pad image
#Zero padd input image to the nearest power of 2 rows and columns
#to make sure that the FFT and IFFT function down the line can opperate correctly
def zero_padding(image):
    
    #assert function is used to make sure the input length is divisible by 2
    assert len(image.shape) == 2

    #defining the size of the input, M = rows and N = columns
    M, N = image.shape

    #defining parameters of zero padding
    P = 2*M - 1
    Q = 2*N - 1

    #while loop statment repeatedly executes a target statement as long as a given condition is true
    #the & operator copies a bit to the results if it exists in both operands
    while P&(P-1):
        #assignment operator that stands for P = P + 1
        P+=1
    while Q&(Q-1):
        #assignment operator that stands for Q = Q + 1
        Q+=1

    #returns a new array of given shape, filled with zeros
    padded_image = np.zeros((P, Q))
    #puts original values in top left quadrent of zero array
    padded_image[:M,:N] = image[:,:]

    return padded_image

#Cooley-Turkey decimation-in-time radix-2 algorithm. needs the input x's length to be of the power of 2 to compute the FFT in the next step
#the inverse=False is used in the Hibernate cascade option, 
#where an inverse keyword is used in a hibernate mapping file for maintaining the relationship of the owner and associated non-owner object
def CT_bit_fft(x, inverse=False):

    #defining bit reversal permutation
    #returns the integer that is the reverse of the smallest 'bits' bits of the integer xi
    def br_permutation(xi, bits):
        y = 0
        
        #calculating a number y having set bits in range of i to bits
        #the << operator means the left operands values is moved left by the number of bits specified by the right operand
        #the >>= performs bitwise right shift and assigns value to the left operand 
        for i in range(bits):
            y = (y << 1) | (xi & 1)
            xi >>= 1
        return y

    #defining variable K
    K = x.shape[0]
    
    #returns the number of bits necessary to represent an integer in binary, excluding the sign and leading zeros
    # levels = log2(n)
    levels = K.bit_length()-1   

    #the != operate states that is values of two operands are not equal, then condition becomes true
    #important to compute the FFT down the line
    if 2 ** levels != K:
        raise ValueError("size not in power of 2")

    #defining variable B, using the Hibernate cascade
    B = (2j if inverse else -2j) * np.pi / K
    
    #defining variable W by using arrange function, to return evenly spaced values within a given interval
    #and using the exp function to calculate the exponential of all the elements in the input array
    W = np.exp(np.arange(K//2) * B)

    #x is copied with bit-reversed permutation
    x = [x[br_permutation(i, levels)] for i in range(K)]

    #Radix-2 decimation-in-time FFT is simplest and most common form of Cooley-Tuey algorithm
    #Radix-2 DIT divides a DFT of size K into two interleaved DFTs of size K/2 with each recursive stage
    size = 2
    while size <= K:
        #floor division
        h_size = size // 2
        t_step = K // size

        for i in range(0, K, size):
            k = 0
            for j in range(i, i + h_size):
                t = x[j + h_size] * W[k]
                x[j + h_size] = x[j] - t
                x[j] += t
                k += t_step

        #multiplies right operand with left operand and assign the result to left operand
        size *= 2

    return np.asarray(x)

#defining function for FFT2D
def FFT2D(image):
    M, N = image.shape

    #returns an array of zeros with the same shape as the given array
    FFT_result = np.zeros_like(image, dtype=complex)

    #implementing Cooley-Turkey function to the rows and columns
    for i in range(M):
        FFT_result[i,:] = CT_bit_fft(image[i,:])

    for j in range(N):
        FFT_result[:, j] = CT_bit_fft(FFT_result[:, j])

    return FFT_result

#defining FFT2D shift function to shift the origin to the center -> i.e. works same as multiplying the input array times (-1)^(x+y)
#was having trouble multiplying a complex array times (-1)^(x+y) and found out that this worked much better
#and still gave good results
def FFT2D_shift(fft):
    rows, cols = fft.shape
    tmp = np.zeros_like(fft)
    ret = np.zeros_like(fft)

    for i in range(rows):
        for j in range(cols):
            #the % operator divides left hand operand by right hand operand and returns remainder
            index = (cols//2 + j) % cols
            tmp[i, index] = fft[i, j]

    for j in range(cols):
        for i in range(rows):
            index = (rows//2 + i) % rows
            ret[index, j] = tmp[i, j]

    return ret

#defining ideal low pass filter function
def ILPF(mask_size, d_0):
    u = mask_size.shape[0]
    v = mask_size.shape[1]
    mask = np.ones((u,v))
    center1 = u/2
    center2 = v/2
    for i in range(1,u):
        for j in range(1,v):
            r1 = (i - center1)**2 + (j - center2)**2
            r = math.sqrt(r1)
            if r > d_0:
                mask[i,j] = 0.0
            elif r <= d_0:
               mask[i,j] = 1
    ilpf = Image.fromarray(mask)
    return ilpf

#defining the 2D inverse FFT function
def IFFT_2D(fu):
    
    #have to include a function that computes the 1D DFT of an input array, first the column component and then the row component
    #will be used in the main IFFT_2D function
    def DFT_1D(fx):
        
        #convert input into an complex array 
        fx = np.asarray(fx, dtype=complex)
        M = fx.shape[0]
        fu = fx.copy()

        for i in range(M):
            u = i
            sum = 0
            for j in range(M):
                x = j
                t = fx[x]*np.exp(-2j*np.pi*x*u*np.divide(1, M, dtype=complex))
                sum += t
                fu[u] = sum
                
        return fu
    
    #define 1D FFT, to be used later, first when taking the column component and then when taking the row component
    def FFT_1D(fx):
        fx = np.asarray(fx, dtype=complex)
        M = fx.shape[0]
        min_Divide_Size = 4

        #the % operator divides left hand operand by right hand operand and returns remainder
        #the != operator stands for if values of two operands are not equal, then the condition becomes true
        #this is why it is important to have the input array from previous steps be as a power of 2 
        if M % 2 != 0:
            raise ValueError("the input size must be 2^n")

        if M <= min_Divide_Size:
            return DFT_1D(fx)
        else:
            fx_even = FFT_1D(fx[::2])  # compute the even part
            fx_odd = FFT_1D(fx[1::2])  # compute the odd part
            C = np.exp(-2j * np.pi * np.arange(M) / M)
            
            f_u = fx_even + fx_odd * C[:M//2]
            f_u1 = fx_even + fx_odd * C[M//2:]

            #join a sequence of arrays along an existing axis
            fu = np.concatenate([f_u, f_u1])

        return fu
    
    #defining 1D IFFT, to be used later, first when taking the column component and then when taking the row component
    def IFFT_1D(fu):
        fu = np.asarray(fu, dtype=complex)
        
        #returns the complex conjugate, element-wise
        #the complex conjugate of a complex number is obtained by changing the sign of its imaginary part
        fu_conjugate = np.conjugate(fu)

        #calling the previously defined 1D FFT function
        fx = FFT_1D(fu_conjugate)

        fx = np.conjugate(fx)
        fx = fx / fu.shape[0]

        return fx
    
    height, width = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(height):
            fx[i, :] = IFFT_1D(fu[i, :])

        for i in range(width):
            fx[:, i] = IFFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = IFFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx

#calling input image
img = cv2.imread('C:/Users/Berto/p3input.jpg', 0)

#calling zero_padding function to add zeros to the top right, bottom left, and bottom right quadrents
z_pad = zero_padding(img)

#calling the 2D fast fourier transfrom function
fft2d = FFT2D(z_pad)

#calling the 2D FFT shift function -> i.e. the (-1)^(x+y) -> to move the origin from the top left to the middle of the array
fshift = FFT2D_shift(fft2d)
#to calculate the magnitude of the fast fourier transform so that i can plot it to see what it looks like
spectrum1 = np.log10(np.absolute(fshift) + np.ones_like(z_pad))

#calling the ideal low pass function where the number after the comma is the cutoff frequency
ilpf_mask = ILPF(z_pad, 20)
#multiplying the ideal low pass mask with the shifted fast fourier transfrom
con = ilpf_mask * fshift

#calling the 2D inverse fast fourier transform function
ifft0 = IFFT_2D(con)
#calculating magnitude of unshifted 2D IFFT to plot what it looks like
unshifted = cv2.magnitude(ifft0[:, :], ifft0[:, :])

#extracting the M*N sub-region from the top-left quadrant of the 2D IFFT
ifft = ifft0[:256, :256]
#calculating the magnitude of the 
spectrum2 = cv2.magnitude(ifft[:, :], ifft[:, :])

#plotting some various images
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(spectrum1, cmap='gray')
ax2.title.set_text('FFT using code')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(unshifted, cmap='gray')
ax3.title.set_text('unshifted IFFT')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(spectrum2, cmap='gray')
ax4.title.set_text('Shifted IFFT using code, d_0 = 80')
plt.show()

mpimg.imsave("d_0 = 20.jpg", spectrum2)