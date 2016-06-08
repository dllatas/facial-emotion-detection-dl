from __future__ import division
import os
import random
import numpy as np

# Random guess for label
def get_label_prob(dict, label):
	dict_len = len(dict)
	true_counter = 0
	for d in dict:
		if d == label:
			true_counter = true_counter + 1
	return true_counter/dict_len

def guess_random_label(dict, labels):
	dict_len = len(dict)
	true_counter = 0
	for d in dict:
		if d == pick_random_label(labels):
			true_counter = true_counter + 1
	return true_counter/dict_len

def pick_random_label(labels):
	return random.choice(labels)

def get_label_dict(path):
	label_dict = []
	for root, dirs, files in os.walk(path, True):
		for name in files:
			f = open(os.path.join(root, name), 'r')
			label_dict.append(int(float(f.read())))
	return label_dict
	
def main(argv=None):  # pylint: disable=unused-argument
	label_path = "/home/neo/projects/deepLearning/data/label/"
	labels = [1,2,3,4,5,6]
	label = 6
	steps = 1000
	result = []
	for i in xrange(steps):
		result.append(guess_random_label(get_label_dict(label_path), labels))
	print round(np.mean(result),4)
	"""
	print get_label_prob(get_label_dict(label_path), label)
	"""
	
if __name__ == '__main__':
	main()