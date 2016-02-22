import tensorflow as tf
import os

labelPath = "/home/neo/projects/deepLearning/data/labelTest/"
labelFormat = "*.txt"
imagePath = "/home/neo/projects/deepLearning/data/imageTest/"
imageFormat = "*.png"

# Global variables

label_container = []
image_container = []
num_epochs = 1

# Input labelTest

labelQueue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(labelPath, labelFormat)), num_epochs=num_epochs)
labelReader = tf.TextLineReader()
labelKey, labelValue = labelReader.read(labelQueue)

# Input imageTest

imageQueue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(imagePath, imageFormat)), num_epochs=num_epochs)
imageReader = tf.WholeFileReader()
imageKey, imageValue = imageReader.read(imageQueue)
imageDecode = tf.image.decode_png(imageValue)

# Running !!

with tf.Session() as sess:
    # Initialize the variables define ("Read more about it !!")
    tf.initialize_all_variables().run()
    # Start to populate the label queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Execute the label section of the graph
    # Save the labels on a variables
    try:
        while not coord.should_stop():
            labelTensor = sess.run([labelKey, labelValue])
            label_container.append([labelTensor[0], int(float(labelTensor[1]))])
    except tf.errors.OutOfRangeError:
        print('Done with the label queue')
    finally:
        # When done, ask the threads to stop.
        # coord.request_stop()
        print label_container
    # Execute the image section of the graph
    # Use the Save the labels on a variables
    try:
        while not coord.should_stop():
            # Get an image tensor and print its value.
            imageTensor = sess.run([imageKey, imageDecode])
            image_container.append([imageTensor[0],imageTensor[1],label_container])
    except tf.errors.OutOfRangeError:
        print('Done with the label queue')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        print(image_container)
        print("long gone !!!")
    # Shutdown the queue coordinator.
    # coord.request_stop()
    coord.join(threads)
