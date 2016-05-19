import os
import numpy as np
from PIL import Image

def crop_image(path, witdh, height):
	offset_h = 10
	for root, dirs, files in os.walk(path, True):
		for name in files:
			image = Image.open(os.path.join(root, name))
			w, h = image.size
			offset_w = (w - witdh) / 2
			image.crop((0 + offset_w, 0 + offset_h, witdh + offset_w, height + offset_h)).save(os.path.join(root, name))


def convert_to_greyscale(path):
	for root, dirs, files in os.walk(path, True):
		for name in files:
			image = Image.open(os.path.join(root, name))
			#image.convert('LA').save(os.path.join(root, name))
			image.convert('L').save(os.path.join(root, name))

def rename_filename(filename):
	#end_position = 8
	end_position = 17
	return filename[:end_position]

def search_label_dictionary(label_dict, filename):
	name = rename_filename(filename)
 	for index, value in label_dict:
 		if index == name:
 			return value

def generate_label_dictionary(path):
	label = []
	for root, dirs, files in os.walk(path, True):
		for name in files:
			f = open(os.path.join(root, name), 'r')
			label.append([rename_filename(name), int(float(f.read()))])
	return label

def set_record_size(label, witdh, height, channel):
	return label + (witdh * height * channel)

def generate_bin(path, total_images, record_size, label_dict):
	result = np.empty([total_images, record_size], dtype=np.uint8)
	i = 0
	for root, dirs, files in os.walk(path, True):
		for name in files:
			label = search_label_dictionary(label_dict, name)
			if label is not None:
				image = Image.open(os.path.join(root, name))
				image_modifed = np.array(image)
				grey = image_modifed[:].flatten()
				result[i] = np.array(list([label]) + list(grey), np.uint8)
				i = i + 1
	result.tofile("/home/neo/projects/deepLearning/data/kh.bin")

def main(argv=None):  # pylint: disable=unused-argument
	label_path = "/home/neo/projects/deepLearning/data/label/"
	image_path = "/home/neo/projects/deepLearning/data/image/"
	#total_images = 4895
	total_images = 327
	witdh = 640
	height = 480
	channel = 1
	label = 1
	#crop_image(image_path, witdh, height)
	#convert_to_greyscale(image_path)
	label_dict = generate_label_dictionary(label_path)
	record_size = set_record_size(label, witdh, height, channel)
	generate_bin(image_path, total_images, record_size, label_dict)
	
if __name__ == '__main__':
	main()