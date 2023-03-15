# mean filter kernel
kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9) # 3x3

#Gaussian kernel
size=5
sigma=3
center=int(size/2)
kernel=np.zeros((size,size))
for i in range(size):
	for j in range(size):
          kernel[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-center)**2+(j-center)**2)/(2*sigma**2))
kernel=kernel/np.sum(kernel)	#Normalize values so that sum is 1.0

#dimensions of the image and the kernel
image_height, image_width = imageGray.shape
kernel_height, kernel_width = ______________________

#Padding
imagePadded = np.zeros((image_height+kernel_height-1,________________________))
for i in range(image_height):
	for j in range(image_width):
		imagePadded[i+int((kernel_height-1)/2), j+_____________________] = imageGray[i,j]  #copy Image to padded array


#correlation
for i in range(________):
	for j in range(image_width):
		window = imagePadded[____________________, j:j+kernel_width]
		imageGray[i,j] = np.sum(window*kernel)  #numpy does element-wise multiplication on arrays

#convolution
		#np.flip(kernel)  # flips horizontally and vertically
		#correlation

#low pass filter
	#do either convolution or correlation using Gaussian kernel
	#since Gaussian kernel is all-axis symmetric, either correlation or convolution can be used
	
#high pass filter
	# original image - low pass image

#merge two images (low pass and high pass)
	alpha*Image1 + (1-alpha)Image2

