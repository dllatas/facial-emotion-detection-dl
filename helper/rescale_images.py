import os
import cv2
import numpy as np

# 290 pixels; approx mean for all the images
# Reduce to 32 pixels image

def get_image_mean(path):
	image_dict = []
	for root, dirs, files in os.walk(path, True):
		for name in files:
			image = cv2.imread(os.path.join(root, name))
			print "File " + root + name + " has height, width, channel: " + str(image.shape)
			image_dict.append(image.shape[0])
	return np.mean(image_dict)

def save_image(image, dest_path, filename):
	cv2.imwrite(os.path.join(dest_path, filename), image)

def resize_single_image(path, name, dest_path, dimension):
	resized = resize_image(path, name, dimension)
	save_image(resized, dest_path, name)	

def resize_multiple_images(path, dest_path, dimension):
	for root, dirs, files in os.walk(path, True):
		for name in files:
			resized = resize_image(root, name, dimension)
			save_image(resized, dest_path, name)

def set_image_dimension(image, dimension, squared):
	if squared:
		dim = (dimension, dimension)
	else:
		ratio = float(dimension) / image.shape[1]
		dim = (dimension, int(image.shape[0] * ratio))
	return dim

def resize_image(path, filename, dimension):
	image = cv2.imread(os.path.join(path, filename))
	dim = set_image_dimension(image, dimension, True)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def main(argv=None):  # pylint: disable=unused-argument
	image_path = "/home/neo/projects/deepLearning/data/crop_faces/"
	dest_path = "/home/neo/projects/deepLearning/data/resize_faces/"
	name = "S005_001_00000007.png"
	dimension = 32
	resize_multiple_images(image_path, dest_path, dimension)
	#resize_single_image(image_path, name, dest_path, dimension)
	#get_image_mean(dest_path)

if __name__ == '__main__':
	main()