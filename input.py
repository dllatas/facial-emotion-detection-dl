import tensorflow as tf
import os

#LABEL_PATH = "/home/neo/projects/deepLearning/data/labelTest/"
#IMAGE_PATH = "/home/neo/projects/deepLearning/data/imageTest/"
LABEL_PATH = "/home/neo/projects/deepLearning/data/label/"
IMAGE_PATH = "/home/neo/projects/deepLearning/data/image/"
LABEL_SUFIX = "_emotion"
LABEL_FORMAT = "*.txt"
IMAGE_FORMAT = "*.png"
CHAR_COLON = ":"
NUM_EPOCHS = 1
LABEL = []

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
#NUM_EPOCHS_PER_DECAY = 350.0
NUM_EPOCHS_PER_DECAY = 35.0
NUM_CLASSES = 7
INITIAL_LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.1
MOVING_AVERAGE_DECAY = 0.9999

# Basic model parameters.
FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size', 48, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('train_dir', '/home/neo/projects/deepLearning/log', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', '/home/neo/projects/deepLearning/log/test', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/neo/projects/deepLearning/log', """Directory where to read model checkpoints.""")
# tf.app.flags.DEFINE_integer('max_steps', 1 000 000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_examples', 1000, """Number of examples to run.""")


def search_label(filename):
    for lab in LABEL:
        # if lab[0] == filename:
        #print lab[0][:8], filename[:8]
        if lab[0][:8] == filename[:8]:
            return lab[1]

def rename_image_filename(image):
    with tf.Session() as ppro:
        # Initialize the variables define ("Read more about it !!")
        tf.initialize_all_variables().run()
        # Start to populate the label queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Execute the image section of the graph
        imageTensor = ppro.run([image])
        key = imageTensor[0]
        # Shutdown the queue coordinator.
        coord.request_stop()
        coord.join(threads)
        return key[len(IMAGE_PATH):-len(IMAGE_FORMAT)+1]

def rename_label_filename(label):
    return label[:-(len(LABEL_SUFIX)+len(LABEL_FORMAT)-1)]

def generate_label_dict(path):
    for root, dirs, files in os.walk(path, True):
        for name in files:
            f = open(os.path.join(root, name), 'r')
            LABEL.append([rename_label_filename(name), int(float(f.read()))])
    return LABEL

def generate_train_batch(label, image, batch_size=FLAGS.batch_size):
    num_preprocess_threads = 1
    min_fraction_of_examples_in_queue = 0.5
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        # capacity=4,
        min_after_dequeue=min_queue_examples
        # min_after_dequeue=1
        )
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

def read_input(image_queue):
    # Read the images and generate the decode from PNG image
    imageReader = tf.WholeFileReader()
    image_key, image_value = imageReader.read(image_queue)
    image_decode = tf.image.decode_png(image_value, channels=1)
    image_decode = tf.cast(image_decode, tf.float32)
    # Preprocess data
    image_key = rename_image_filename(image_key)    # rename image filename 
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
    height = 96
    width = 96
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
    return generate_train_batch(record.label, float_image)

def get_input(label_path, label_format, image_path, image_format):
    generate_label_dict(label_path)
    image_queue = tf.train.string_input_producer(tf.train.match_filenames_once(os.path.join(image_path, image_format)), num_epochs=NUM_EPOCHS)
    return read_input(image_queue)