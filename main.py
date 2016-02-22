import tensorflow as tf
import os

# Define paths to load labels and images
labelPath = "/home/neo/projects/deepLearning/data/labelTest/"
labelFormat = "*.txt"
labelSufix = "_emotion"
imagePath = "/home/neo/projects/deepLearning/data/imageTest/"
imageFormat = "*.png"

# Global variables
char_colon = ":"
label_container = []
image_container = []
num_epochs = 1
filename_length = 17

# Input labelTest
labelQueue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(labelPath, labelFormat)), num_epochs=num_epochs)
labelReader = tf.TextLineReader()
labelKey, labelValue = labelReader.read(labelQueue)

# Input imageTest
imageQueue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(imagePath, imageFormat)), num_epochs=num_epochs)
imageReader = tf.WholeFileReader()
imageKey, imageValue = imageReader.read(imageQueue)
imageDecode = tf.image.decode_png(imageValue)

# Business logic
def format_label(label):
    labelResult = []
    for i in xrange(len(label)):
        labelResult.append([label[i][0][len(labelPath):(label[i][0].index(char_colon)-len(labelFormat)+1-len(labelSufix))], label[i][1]])
    return labelResult

def format_image(image):
    imageResult = []
    for i in xrange(len(image)):
        imageResult.append([image[i][0][len(imagePath):-len(labelFormat)+1], image[i][1]])
    return imageResult

def match_label_with_images(label, image):
    train = []
    for i in xrange(len(label)):
        for j in xrange(len(image)):
            if label[i][0]==image[j][0]:
                train.append([image[j][0], label[i][1], image[j][1]])
    return train

# Graph execution
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
        pass
        # print('Done with the label queue')
    finally:
        # When done, let's format the label's name
        label_container = format_label(label_container)
    # Execute the image section of the graph
    # Use the Save the labels on a variables
    try:
        while not coord.should_stop():
            # Get an image tensor and print its value.
            imageTensor = sess.run([imageKey, imageDecode])
            image_container.append([imageTensor[0],imageTensor[1]])
    except tf.errors.OutOfRangeError:
        print('Done with the image queue')
    finally:
        # When done, ask the threads to stop.
        image_container = format_image(image_container)
        train = match_label_with_images(label_container, image_container)
        print len(train)
        coord.request_stop()
    # Shutdown the queue coordinator.
    # coord.request_stop()
    coord.join(threads)
