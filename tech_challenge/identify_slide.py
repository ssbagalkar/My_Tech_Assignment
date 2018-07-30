import warnings
import cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy.matlib
import sklearn
import os
import glob
import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
# diagnose function using HSV value thresholding
##### This part uses only HSV hard-coded values to differentiate between 2 tissues #######

def extract_mean_hsv(input_img):
	gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	ret, img_bw = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	image_complement = cv2.bitwise_not(img_bw)
	repmat_binary = np.repeat(image_complement[:, :, np.newaxis], 3, axis=2)
	rgb_zeroed = cv2.bitwise_and(input_img, repmat_binary)
	hsv = cv2.cvtColor(rgb_zeroed, cv2.COLOR_BGR2HSV)
	hue = hsv[:, :, 0].flatten()
	saturation = hsv[:, :, 1].flatten()
	value = hsv[:, :, 2].flatten()
	mean_hue = (np.true_divide(hue.sum(0), (hue != 0).sum(0))) * 2
	mean_sat = (np.true_divide(saturation.sum(0), (saturation != 0).sum(0)))
	mean_value = (np.true_divide(value.sum(0), (value != 0).sum(0)))
	return mean_hue, mean_sat, mean_value
	
def diagnose_cv(input_img):
	[mean_hue, mean_sat, mean_value] = extract_mean_hsv(input_img)
	if 300<=mean_hue<=330 and mean_sat <= 127.5 and mean_value>=153:
		print("A")
	else:
		print("B")

### This part uses HSV values as a feature vector to train SVM ###
def extract_features(images_array):
	features=[]
	if images_array.size > 10:
		feature_image = cv2.cvtColor(images_array, cv2.COLOR_RGB2HSV)
		hist_features = extract_mean_hsv(feature_image)
		features.append(hist_features)
	else:
		for image in images_array:
			feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			hist_features = extract_mean_hsv(feature_image)
			features.append(hist_features)
	return np.array(features)


def collect_data():
	pathology_A_images=[]
	pathology_B_images=[]
	
	base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
	image_file_a = glob.glob(base_directory + "\\pathology_A\\*.PNG")
	image_file_b = glob.glob(base_directory + "\\pathology_B\\*.PNG")
	for path_a_image, path_b_image in zip(image_file_a, image_file_b):
		image_a = cv2.imread(path_a_image)
		image_b = cv2.imread(path_b_image)
		pathology_A_images.append(image_a)
		pathology_B_images.append(image_b)

	a_data = np.array(pathology_A_images)
	b_data = np.array(pathology_B_images)

	return a_data, b_data

def train_model(path_a_features, path_b_features):
	X = np.vstack((path_a_features, path_b_features)).astype(np.float64)
	# Fit a per-column scalar
	# X_scaler = StandardScaler().fit(X)
	#
	# Apply the scalar to X
	# scaled_X = X_scaler.transform(X)

	# define labels
	y = np.hstack((np.ones(len(path_a_features)), np.zeros(len(path_b_features))))

	# Use a random seeding
	rand_state = np.random.randint(0, 10)

	# split data set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state)

	# Using a SVM with polynomial kernel
	# print("Training started")
	model = SVC(kernel='poly')
	clf = model.fit(X_train, y_train)
	# print("Training ended")

	filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) + "\\svm_model_poly_trained.sav"
	joblib.dump(clf, filename)
	# print("Model saved")
	# check score of SVC
	# print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

def diagnose_with_svm(input_image):
	features = extract_features(input_image)
	filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) + "\\svm_model_poly_trained.sav"
	clf = joblib.load(filename)
	prediction = clf.predict(features)
	if prediction == 0:
		print("B")
	else:
		print("A")

if __name__ == '__main__':
	user_entered_path = sys.argv[1]
	if not os.path.exists(user_entered_path):
		print("Path Invalid. Please enter a valid file path")
	else:
		input_img = cv2.imread(user_entered_path)
		# Give option to run only using hard coded thresholds or using ML algorithm
		print("Process using ML algorithm or using only CV methods ? Press 1 for ML or 0 for just CV")
		load_model_choice = int(input())
		if load_model_choice == 0:
			diagnose_cv(input_img)
		else:
			[a_data, b_data] = collect_data()
			path_a_features = extract_features(a_data)
			path_b_features = extract_features(b_data)
			train_model(path_a_features, path_b_features)
			diagnose_with_svm(input_img)