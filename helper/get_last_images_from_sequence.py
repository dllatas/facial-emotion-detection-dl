import os
import collections

# Get the last x images from all the sequences

def move_image_sequences(path, sequences_to_move, dest_path):
	for root, dirs, files in os.walk(path, True):
		for name in files:
			if name in sequences_to_move:
				os.rename(os.path.join(root, name), os.path.join(dest_path, name))

def flatten_list(list_to_flat):
    for el in list_to_flat:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten_list(el):
                yield sub
        else:
            yield el

def generate_image_value(sequence_value, image_value, scope, image_value_length, file_format):
	filenames = []
	border = int(image_value) - scope
	for i in xrange(border, int(image_value)):
		filenames.append(sequence_value + "_" + str(i + 1).zfill(image_value_length) + file_format)
	return filenames

def get_sequence_from_filename(filename):
	return filename[:8]

def get_image_from_filename(filename):
	return filename[9:17]

def generate_sequence_dictionary(label_path, scope, image_value_length, file_format):
	filenames = []
	for root, dirs, files in os.walk(label_path, True):
		for name in files:
			filenames.append(generate_image_value(get_sequence_from_filename(name), get_image_from_filename(name), scope, image_value_length, file_format))
	return list(flatten_list(filenames))

def main(argv=None):  # pylint: disable=unused-argument
	label_path = "/home/neo/projects/deepLearning/data/label/"
	image_path = "/home/neo/projects/deepLearning/data/image/"
	dest_path = "/home/neo/projects/deepLearning/data/image_exp2/"
	scope = 5
	image_value_length = 8
	file_format = ".png"
	sequence_dict = generate_sequence_dictionary(label_path, scope, image_value_length, file_format)
	move_image_sequences(image_path, sequence_dict, dest_path)

if __name__ == '__main__':
	main()