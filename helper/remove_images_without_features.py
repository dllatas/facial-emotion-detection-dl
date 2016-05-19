# Images ending with a range of values between 1 and 3 dont provide 
# with any particular feature for the experiment. That's the reason 
# they will be prunned from the dataset !!
import os

IMAGE_PATH = "/home/neo/projects/deepLearning/data/image/"
TO_MOVE_PATH = "/home/neo/projects/deepLearning/data/unused_images/"

INITIAL_CHAR = 9 
FINAL_CHAR = 17
TO_PRUNE = [1, 2, 3]

for root, dirs, files in os.walk(IMAGE_PATH, True):
	for name in files:
		if int(float(name[INITIAL_CHAR:FINAL_CHAR])) in TO_PRUNE:
			os.rename(os.path.join(IMAGE_PATH, name), os.path.join(TO_MOVE_PATH, name))


"""
PATH = "/home/neo/projects/deepLearning/data"
NAME = "unused_images"

for root, dirs, files in os.walk(TO_MOVE_PATH, True):
	for name in files:
		if name[:len(NAME)] == NAME:
			os.rename(os.path.join(TO_MOVE_PATH, name), os.path.join(TO_MOVE_PATH, name[len(NAME):]))
"""