import tensorflow as tf
import input


# Global constants describing the CIFAR-10 data set.
LABEL_PATH = input.LABEL_PATH
IMAGE_PATH = input.IMAGE_PATH
LABEL_SUFIX = input.LABEL_SUFIX
LABEL_FORMAT = input.LABEL_FORMAT
IMAGE_FORMAT = input.IMAGE_FORMAT

def train():
    with tf.Graph().as_default():
        input.get_input(LABEL_PATH, LABEL_FORMAT, IMAGE_PATH, IMAGE_FORMAT)

def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()
