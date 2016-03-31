import os

# path = "/home/neo/projects/deepLearning/data/extended-cohn-kanade-images/"
path = "/home/neo/projects/deepLearning/data/emotion_labels/"

# rep = "/home/neo/projects/deepLearning/data/image/"
rep = "/home/neo/projects/deepLearning/data/label"

for root, dirs, files in os.walk(path, topdown=True):
    if len(files) > 0:
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(rep, file))
