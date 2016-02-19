import os

path = "/home/neo/projects/deepLearning/data2/"

prefix = "data2"

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_files(dir):
    return [name for name in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, name))]

folder = get_immediate_subdirectories(path)

for i in xrange(len(folder)):
    os.rename(path+folder[i],path+prefix+folder[i])

folder = get_immediate_subdirectories(path)

for i in xrange(len(folder)):
    newPath = os.path.join(path, folder[i])
    files = get_files(newPath)
    for j in xrange(len(files)):
        os.rename(os.path.join(newPath, files[j]),os.path.join(newPath, folder[i] + files[j]))
