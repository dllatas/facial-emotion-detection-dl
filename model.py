import tensorflow as tf
import input

def gather_summary(x):
    tf.histogram_summary('/activations', x)
    tf.scalar_summary('/sparsity', tf.nn.zero_fraction(x))

def set_variable(name, shape, initializer):
     var = tf.get_variable(name, shape, initializer=initializer)
     return var

def set_weight_variable(name, shape, stddev, wd):
    var = set_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight)
    return var

def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = set_weight_variable('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = set_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        gather_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = set_weight_variable('weights', shape=[5, 5, 64, 64],stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = set_variable('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        gather_summary(conv2)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
        weights = set_weight_variable('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = set_variable('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        gather_summary(local3)

    with tf.variable_scope('local4') as scope:
        weights = set_weight_variable('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = set_variable('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        gather_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = set_weight_variable('weights', [192, NUM_CLASSES],stddev=1/192.0, wd=0.0)
        biases = set_variable('biases', [NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        gather_summary(softmax_linear)

    return softmax_linear
