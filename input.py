import tensorflow as tf
import os

LABEL_PATH = "/home/neo/projects/deepLearning/data/labelTest/"
IMAGE_PATH = "/home/neo/projects/deepLearning/data/imageTest/"

LABEL_SUFIX = "_emotion"
LABEL_FORMAT = "*.txt"
IMAGE_FORMAT = "*.png"

CHAR_COLON = ":"
NUM_EPOCHS = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")

def format_label(label):
    labelResult = []
    for i in xrange(len(label)):
        labelResult.append([label[i][0][len(LABEL_PATH):(label[i][0].index(CHAR_COLON)-len(LABEL_FORMAT)+1-len(LABEL_SUFIX))], label[i][1]])
    return labelResult

def format_image(image):
    imageResult = []
    for i in xrange(len(image)):
        imageResult.append([image[i][0][len(IMAGE_PATH):-len(IMAGE_FORMAT)+1], image[i][1]])
    return imageResult

def match_label_with_images(label, image):
    train = []
    for i in xrange(len(label)):
        for j in xrange(len(image)):
            if label[i][0]==image[j][0]:
                train.append([image[j][0], label[i][1], image[j][1]])
                break
    return train

def process_input(label_key, label_value, image_key, image_decode):
    label_container = []
    image_container = []
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
                labelTensor = sess.run([label_key, label_value])
                label_container.append([labelTensor[0], int(float(labelTensor[1]))])
        except tf.errors.OutOfRangeError:
            pass
        finally:
            # When done, let's format the label's name
            label_container = format_label(label_container)
        # Execute the image section of the graph
        # Use the Save the labels on a variables
        try:
            while not coord.should_stop():
                # Get an image tensor and print its value.
                imageTensor = sess.run([image_key, image_decode])
                image_container.append([imageTensor[0],imageTensor[1]])
        except tf.errors.OutOfRangeError:
            pass
            # print('Done with the image queue')
        finally:
            # When done, ask the threads to stop.
            image_container = format_image(image_container)
            train = match_label_with_images(label_container, image_container)
            coord.request_stop()
        # Shutdown the queue coordinator.
        # coord.request_stop()
        coord.join(threads)
        return train

def generate_train_batch(label, image, batch_size=FLAGS.batch_size):
    num_preprocess_threads = 16
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
        )
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

def distort_input(record):
    label = []
    image = []
    for i in xrange(len(record)):
        record[i][1] = tf.cast(record[i][1], tf.int32)
        record[i][2] = tf.cast(record[i][2], tf.float32)
        label.append(record[i][1])
        image.append(record[i][2])
        # Crop image
        # Flip left to right
        # Modify brightness
        # Modify contrast
        # Apply whitening
    label = tf.pack(label, name="label")
    image = tf.pack(image, name="image")
    return generate_train_batch(label, image)

def read_input(label_queue, image_queue):
    # Read the labels
    labelReader = tf.TextLineReader()
    label_key, label_value = labelReader.read(label_queue)
    # Read the labels and generate the decode from PNG image
    imageReader = tf.WholeFileReader()
    image_key, image_value = imageReader.read(image_queue)
    image_decode = tf.image.decode_png(image_value)
    # Preprocess data
    record = process_input(label_key, label_value, image_key, image_decode)
    distorted_record = distort_input(record)
    return distorted_record

def get_input(label_path, label_format, image_path, image_format):
    label_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(label_path, label_format)), num_epochs=NUM_EPOCHS)
    image_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(image_path, image_format)), num_epochs=NUM_EPOCHS)
    read_input(label_queue, image_queue)
