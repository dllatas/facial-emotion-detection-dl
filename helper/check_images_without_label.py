import os

LABEL_PATH = "/home/neo/projects/deepLearning/data/label/"
IMAGE_PATH = "/home/neo/projects/deepLearning/data/image/"
TO_MOVE_PATH = "/home/neo/projects/deepLearning/data/gg/"
LABEL_SUFIX = "_emotion"
LABEL_FORMAT = "*.txt"
IMAGE_FORMAT = "*.png"
CHAR_COLON = ":"

LABEL = []
IMAGE = []
TO_MOVE = []

for root, dirs, files in os.walk(LABEL_PATH, True):
	for name in files:
		# LABEL.append(name[:-(len(LABEL_SUFIX)+len(LABEL_FORMAT)-1)])
		LABEL.append(name[:8])

for root, dirs, files in os.walk(IMAGE_PATH, True):
	for name in files:
		# LABEL.append(name[:-(len(LABEL_SUFIX)+len(LABEL_FORMAT)-1)])
		IMAGE.append(name[:8])

LABEL = sorted(set(LABEL))
IMAGE = sorted(set(IMAGE))

for i in range(len(IMAGE)):
	found = False
	for j in range(len(LABEL)):
		if IMAGE[i] == LABEL[j]:
			found = True
			break
	if found == False:
		TO_MOVE.append(IMAGE[i])

for root, dirs, files in os.walk(IMAGE_PATH, True):
	for name in files:
		for k in range(len(TO_MOVE)):
			if name[:8] == TO_MOVE[k]:
				os.rename(IMAGE_PATH + name , TO_MOVE_PATH + name)

'''
S005_001_00000001
S005_001_00000010
S005_001_00000011
S005_001_00000011_emotion
Compare the first 8 digits of the image, if they are similar then assign the label

Cuales imagenes no tienen label ????

327 labels for 10709 images
327 labels for 593 image groups

'''