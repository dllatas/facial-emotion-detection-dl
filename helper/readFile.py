import os

LABEL_PATH = "/home/neo/projects/deepLearning/data/labelTest/"
CHAR_COLON = ":"
LABEL_SUFIX = "_emotion"
LABEL_FORMAT = "*.txt"
IMAGE_PATH = "/home/neo/projects/deepLearning/data/imageTest/"
IMAGE_FORMAT = "*.png"


label = []

def format_label2(label):
    return label[:-(len(LABEL_SUFIX)+len(LABEL_FORMAT)-1)]

for root, dirs, files in os.walk(LABEL_PATH, True):
    for name in files:
        f = open(os.path.join(root, name), 'r')
        label.append([format_label2(name), name, int(float(f.read()))])

for lab in label:
    print lab[0], lab[1], lab[2]

image = "/home/neo/projects/deepLearning/data/imageTest/S005_001_00000001.png"

print image[len(IMAGE_PATH):-len(IMAGE_FORMAT)+1]
