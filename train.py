import tensorflow as tf
import input
import model


# Global constants describing the CIFAR-10 data set.
LABEL_PATH = input.LABEL_PATH
IMAGE_PATH = input.IMAGE_PATH
LABEL_SUFIX = input.LABEL_SUFIX
LABEL_FORMAT = input.LABEL_FORMAT
IMAGE_FORMAT = input.IMAGE_FORMAT

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        image, label = input.get_input(LABEL_PATH, LABEL_FORMAT, IMAGE_PATH, IMAGE_FORMAT)
        logits = model.inference(image)


def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()
