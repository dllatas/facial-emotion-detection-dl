import os

# Move images belonging to a particular emotion to a new path

def get_ck_emotion(emotion):
	emotions = {'angry': 1, 'contempt': 2, 'disgust': 3, 'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7}
	return emotions.get(emotion)

def get_sequence_from_filename(filename):
	return filename[:8]

def move_image_sequences(path, sequences_to_move, dest_path):
	for root, dirs, files in os.walk(path, True):
		for name in files:
			if get_sequence_from_filename(name) in sequences_to_move:
				os.rename(os.path.join(root, name), os.path.join(dest_path, name))

def get_emotion_sequences(path, emotion):
	sequence = []
	for root, dirs, files in os.walk(path, True):
		for name in files:
			label = open(os.path.join(root, name), "r")
			if int(float(label.read())) == emotion:
				sequence.append(get_sequence_from_filename(name))
	return sequence

def main(argv=None):  # pylint: disable=unused-argument
	label_path = "/home/neo/projects/deepLearning/data/label/"
	image_path = "/home/neo/projects/deepLearning/data/image/"
	dest_path = "/home/neo/projects/deepLearning/data/ck_contempt/"
	emotion_sequence = get_emotion_sequences(label_path, get_ck_emotion("contempt"))
	move_image_sequences(image_path, emotion_sequence, dest_path)

if __name__ == '__main__':
	main()