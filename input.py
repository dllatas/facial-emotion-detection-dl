import tensorflow as tf
import os

LABEL_PATH = "/home/neo/projects/deepLearning/data/labelTest/"
IMAGE_PATH = "/home/neo/projects/deepLearning/data/imageTest/"
LABEL_SUFIX = "_emotion"
LABEL_FORMAT = "*.txt"
IMAGE_FORMAT = "*.png"
CHAR_COLON = ":"
NUM_EPOCHS = 1
LABEL = []

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4
#NUM_EPOCHS_PER_DECAY = 350.0
NUM_EPOCHS_PER_DECAY = 2.0
NUM_CLASSES = 7
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size', 4, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('train_dir', '/home/neo/projects/deepLearning/log', """Directory where to write event logs and checkpoint.""")
# tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 1000, """Number of batches to run.""")

def search_label(filename):
    for lab in LABEL:
        if lab[0] == filename:
            return lab[1]

def format_image3(image):
    return image[len(IMAGE_PATH):-len(IMAGE_FORMAT)+1]

def format_image2(image):
    with tf.Session() as ppro:
        # Initialize the variables define ("Read more about it !!")
        tf.initialize_all_variables().run()
        # Start to populate the label queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Execute the image section of the graph
        # Use the Save the labels on a variables
        imageTensor = ppro.run([image])
        key = imageTensor[0]
        # Shutdown the queue coordinator.
        coord.request_stop()
        coord.join(threads)
        return format_image3(key)
    # with tf.Session() as sess:
        # image = sess.run(image)
        # print image
        # return image[len(IMAGE_PATH):-len(IMAGE_FORMAT)+1]

def format_label2(label):
    return label[:-(len(LABEL_SUFIX)+len(LABEL_FORMAT)-1)]

def generate_label_dict():
    for root, dirs, files in os.walk(LABEL_PATH, True):
        for name in files:
            f = open(os.path.join(root, name), 'r')
            LABEL.append([format_label2(name), int(float(f.read()))])
    return LABEL

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
    num_preprocess_threads = 1
    min_fraction_of_examples_in_queue = 0.5
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
        # Crop image
        # Flip left to right
        # Modify brightness
        # Modify contrast
        # Apply whitening
        label.append(record[i][1])
        image.append(record[i][2])

    label = tf.pack(label, name="label")
    image = tf.pack(image, name="image")

    return label, image

# def read_input(label_queue, image_queue):
def read_input(image_queue):
    # Read the labels
    # labelReader = tf.TextLineReader()
    # label_key, label_value = labelReader.read(label_queue)
    # label_value = tf.cast(label_value, tf.int32)
    # Read the images and generate the decode from PNG image
    imageReader = tf.WholeFileReader()
    image_key, image_value = imageReader.read(image_queue)
    image_decode = tf.image.decode_png(image_value, channels=1)
    image_decode = tf.cast(image_decode, tf.float32)
    # Preprocess data
    image_key = format_image2(image_key)
    label = search_label(image_key)
    # CREATE OBJECT
    class Record(object):
        pass
    record = Record()
    # Instantiate object
    record.key = image_key
    record.label = tf.cast(label, tf.int32)
    record.image = image_decode
    # PROCESSING IMAGES
    # reshaped_image = tf.cast(record.image, tf.float32)
    # height = 245
    # width = 320
    height = 32
    width = 32
    # Image processing for training the network. Note the many random distortions applied to the image.
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(record.image, [height, width, 1])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Because these operations are not commutative, consider randomizing randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)
    # record = process_input(label_key, label_value, image_key, image_decode)
    # label, image = distort_input(record)
    return generate_train_batch(record.label, float_image)
    # return generate_train_batch(record.label, record.image)
    # return record

def get_input(label_path, label_format, image_path, image_format):
    # label_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(label_path, label_format)), num_epochs=NUM_EPOCHS)
    generate_label_dict()
    image_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(image_path, image_format)), num_epochs=NUM_EPOCHS)
    return read_input(image_queue)
