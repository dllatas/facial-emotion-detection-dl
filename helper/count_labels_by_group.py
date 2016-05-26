from collections import Counter
import os

LABEL_PATH = "/home/neo/projects/deepLearning/data/label/"
LABEL = []

for root, dirs, files in os.walk(LABEL_PATH, True):
	for name in files:
		f = open(os.path.join(root, name), 'r')
		LABEL.append(int(float(f.read())))

print set(Counter(LABEL))