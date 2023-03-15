# lab2 | CMSC 174 | Computer Vision
# Author: John Patrick T. Panizales 

import cv2
import numpy as np

def padding(img, img_dims, kernel_dims):

    #image dimensions
    image_height = img_dims[0]
    image_width = img_dims[1]

    #kernel dimensions
    kernel_height = kernel_dims[0]
    kernel_width = kernel_dims[1]

    imagePadded = np.zeros((image_height+kernel_height-1,image_width+kernel_height-1))
    
    for i in range(image_height):
        for j in range(image_width):
            imagePadded[i+int((kernel_height-1)/2), j+int((kernel_width-1)/2)] = img[i,j]  #copy Image to padded array

    return imagePadded

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # get image dimenstions
    image_dims = img.shape
    img_rows = image_dims[0]
    img_cols = image_dims[1]

    # get kernel dimensions
    kernel_dims = kernel.shape
    kernel_rows = kernel_dims[0]
    kernel_cols = kernel_dims[1]

    # add padding to original image
    imagePadded = padding(img, image_dims,kernel_dims)
    
    # output image
    newImage = np.zeros((imagePadded.shape))

    # correlation
    for i in range(img_rows):
        for j in range(img_cols):
            window = imagePadded[i:i+kernel_rows, j:j+kernel_cols]
            newImage[i+int((kernel_rows-1)/2), j+int((kernel_cols-1)/2)] = np.sum(window*kernel)  #numpy does element-wise multipliccols on arrays

    kernel_center =  kernel_rows//2
    # cut newImage to remove the padding
    newImage = newImage[kernel_center:(newImage.shape[0]-kernel_center)+1, kernel_center:(newImage.shape[1]-kernel_center)+1]

    return newImage

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    flipped_kernel = np.flip(kernel)
    convolution_output = cross_correlation_2d(img,flipped_kernel)
    return convolution_output

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    #Gaussian kernel
    center=int(height/2)
    kernel=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            kernel[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-center)**2+(j-center)**2)/(2*sigma**2))
    
    kernel=kernel/np.sum(kernel)	#Normalize values so that sum is 1.0
    return kernel

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    height = size
    width = size
    kernel = gaussian_blur_kernel_2d(sigma, height, width)
    low_pass_padded = convolve_2d(img,kernel)
    
    return low_pass_padded

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # low_pass_img = low_pass(img, sigma, size)
    # img_padded = padding(img,img.shape, (size,size))
    # high_pass_img = img_padded - low_pass_img

    low_pass_img = low_pass(img, sigma, size)
    high_pass_img = img - low_pass_img

    return high_pass_img

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

# dog cat hybrid test
img1 = cv2.imread("example-images/dog.jpg",0) #low-pass image
img2 = cv2.imread("example-images/cat.jpg",0) # high-pass image 
hybrid_img = create_hybrid_image(img1,img2,5,30,"low",5,30,"high",1/2,2)
cv2.imwrite("cat-dog.jpg", hybrid_img)

# blinking white guy meme test
img1 = cv2.imread("addtl-images/close-eyes.png",0) # high-pass image 
img2 = cv2.imread("addtl-images/open-eyes.png",0) #low-pass image
hybrid_img = create_hybrid_image(img1,img2,5,30,"low",5,30,"high",1/3,2)
cv2.imwrite("open_close.jpg", hybrid_img)

# monkey meme test
img1 = cv2.imread("addtl-images/shrek-human.png",0) # high-pass image 
img2 = cv2.imread("addtl-images/shrek-ogre.png",0) #low-pass image
hybrid_img = create_hybrid_image(img1,img2,5,30,"low",5,30,"high",0.55,3)
cv2.imwrite("shrek.jpg", hybrid_img)

# load base image for testing
# img = cv2.imread("example-images/dog.jpg",0)
# print("original image", img.shape)
# cv2.imwrite("original-image.jpg",img)

# testing parameterse
# sigma = 5
# size = 30

# test for low pass function
# low_pass_img = low_pass(img,sigma,size)
# print("low pass \n", low_pass_img[1,:])
# cv2.imwrite("low-pass-test.jpg",low_pass_img)
# ---

# test for high pass function
# high_pass_img = 3 * high_pass(img,sigma,size)
# print("high pass \n", high_pass_img[1,:])
# cv2.imwrite("high-pass-test.jpg",high_pass_img)
# ---