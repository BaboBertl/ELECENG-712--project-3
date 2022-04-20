# ELECENG-712--project-3
Digital Imaging - Project 3

In this project, you are asked to implement the discrete Fourier transform F(u, v) of an input image f(x, y) of size M*N and then apply the ideal low pass filter H(u, v) to smoothing the image.

Notes:

  1. You need to zero-pad your original image to generate a new image of size P*Q, where P = 2M - 1 and Q = 2N - 1
  2. You need to multiply your original image by (-1)^(x + y) so that the low frequency of F(u, v) is centered at the center of your domain
  3. You are required to use the fast 2D DFT algorithm to implement this project. Otherwise, it'll take you too much time to wait for the DFT results
  4. Please use the image provided as your input
  5. Two output images corresponding to two significantly different cutoff frequencies should be submitted
  6. Please also submit the source code with your output images
  7. If you use Matlab in this project, you cannot simply call the DFT function provided by MATLAB
