import tensorflow as tf
import input

# FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")

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
        kernel = set_weight_variable('weights', shape=[5, 5, 1, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = set_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        gather_summary(conv1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.local_response_normalization(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = set_weight_variable('weights', shape=[5, 5, 64, 64],stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = set_variable('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        gather_summary(conv2)

    norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [input.FLAGS.batch_size, dim])
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
        weights = set_weight_variable('weights', [192, input.NUM_CLASSES],stddev=1/192.0, wd=0.0)
        biases = set_variable('biases', [input.NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        gather_summary(softmax_linear)

    return softmax_linear

def loss(logits, labels):
    # Reshape the labels into a dense Tensor of shape [batch_size, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [input.FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(0, input.FLAGS.batch_size), [input.FLAGS.batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [input.FLAGS.batch_size, input.NUM_CLASSES], 1.0, 0.0)
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, dense_labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summmary to all individual losses and the total loss; do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / input.FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * input.NUM_EPOCHS_PER_DECAY)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(input.INITIAL_LEARNING_RATE, global_step, decay_steps, input.LEARNING_RATE_DECAY_FACTOR, staircase=True)
    tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(input.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
