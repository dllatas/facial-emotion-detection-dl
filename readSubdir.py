import os
import tensorflow as tf

path = "/home/neo/projects/deepLearning/data/extended-cohn-kanade-images"

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

folder = get_immediate_subdirectories(path)

filename2 = []

for i in xrange(len(folder)):
    folder2 = get_immediate_subdirectories(path + "/" + folder[i])
    for j in xrange(len(folder2)):
        # filename_queue = filename_queue + tf.train.string_input_producer(tf.train.match_filenames_once(path + "/" + folder[i] + "/" + folder2[j] + "/*.png"))
        filename = tf.train.match_filenames_once(path + "/" + folder[i] + "/" + folder2[j] + "/*.png")
        # filename2.append(filename)

#filename = tf.train.match_filenames_once(path)
filename_queue = tf.train.string_input_producer(filename)
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
image = tf.image.decode_png(image_file)


with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
