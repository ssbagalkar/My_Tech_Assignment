import cv2
import matplotlib.pyplot as plt
import numpy.matlib
import sklearn
import os
import glob
import numpy as np


base_directory = os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..'))
print(base_directory)
pathology_A_images = []
image_file = glob.glob ("C:\\Users\\saurabh B\\Documents\\Proscia\\My_Tech_Assignment\\pathology_A\\*.PNG")
for myFile in image_file:
	image = cv2.imread (myFile)
	pathology_A_images.append(image)
	# cv2.imshow('A',image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

a_data = np.array(pathology_A_images)

pathology_B_images = []
image_file = glob.glob ("C:\\Users\\saurabh B\\Documents\\Proscia\\My_Tech_Assignment\\pathology_B\\*.PNG")
for myFile in image_file:
	image = cv2.imread (myFile)
	pathology_B_images.append (image)
	# cv2.imshow('B', image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

b_data = np.array(pathology_B_images)

# Process for A
a_hue=[]
a_sat=[]
a_value=[]

for current_img in a_data:
	gray_image = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
	ret,img_bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	image_complement = cv2.bitwise_not(img_bw)
	repmat_binary = np.repeat(image_complement[:, :, np.newaxis], 3, axis=2)
	rgb_zeroed = cv2.bitwise_and(current_img, repmat_binary)
	hsv = cv2.cvtColor(rgb_zeroed, cv2.COLOR_BGR2HSV)
	hue=hsv[:,:,0].flatten()
	saturation = hsv[:,:,1].flatten()
	value = hsv[:,:,2].flatten()
	mean_hue = (np.true_divide(hue.sum(0), (hue != 0).sum(0)))*2
	mean_sat = (np.true_divide(saturation.sum(0), (saturation != 0).sum(0)))/255
	mean_value = (np.true_divide(value.sum(0), (value != 0).sum(0)))/255
	a_hue.append(mean_hue)
	a_sat.append(mean_sat)
	a_value.append(mean_value)
	
for current_img in a_data:
	gray_image = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
	ret,img_bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	image_complement = cv2.bitwise_not(img_bw)
	repmat_binary = np.repeat(image_complement[:, :, np.newaxis], 3, axis=2)
	rgb_zeroed = cv2.bitwise_and(current_img, repmat_binary)
	hsv = cv2.cvtColor(rgb_zeroed, cv2.COLOR_BGR2HSV)
	hue=hsv[:,:,0].flatten()
	saturation = hsv[:,:,1].flatten()
	value = hsv[:,:,2].flatten()
	mean_hue = (np.true_divide(hue.sum(0), (hue != 0).sum(0)))*2
	mean_sat = (np.true_divide(hue.sum(0), (hue != 0).sum(0)))/255
	mean_value = (np.true_divide(hue.sum(0), (hue != 0).sum(0)))/255
	a_hue.append(mean_hue)
	a_sat.append(mean_sat)
	a_value.append(mean_value)