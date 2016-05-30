import os
import cv2
import numpy as np

# 290 pixels; approx mean for all the images

def get_image_mean(path):
	image_dict = []
	for root, dirs, files in os.walk(path, True):
		for name in files:
			image = cv2.imread(os.path.join(root, name))
			print image.shape
			#image_dict.append(image.shape[0])
	#return np.mean(image_dict)

def save_image(image, dest_path, filename):
	cv2.imwrite(os.path.join(dest_path, filename), image)

def get_image_to_resize(path, dest_path):
	for root, dirs, files in os.walk(path, True):
		for name in files:
			image = cv2.imread(os.path.join(root, name))
			dim = set_image_dimension(image, 256)
			resized = resize_image(image, dim)
			save_image(resized, dest_path, name)

def set_image_dimension(image, mean = 290):
	#ratio = float(mean) / image.shape[1]
	#dim = (mean, int(image.shape[0] * ratio))
	dim = (mean, mean)
	return dim

def resize_image(image, dim):
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def main(argv=None):  # pylint: disable=unused-argument
	image_path = "/home/neo/projects/deepLearning/data/crop_faces/"
	dest_path = "/home/neo/projects/deepLearning/data/resize_faces/"
	#get_image_to_resize(image_path, dest_path)
	get_image_mean(dest_path)

if __name__ == '__main__':
	main()