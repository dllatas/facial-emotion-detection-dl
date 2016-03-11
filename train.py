import tensorflow as tf
import numpy as np
import time
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
        loss = model.loss(logits, label)
        train_op = model.train(loss, global_step)
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=input.FLAGS.log_device_placement))
        sess.run(init)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(input.FLAGS.train_dir, graph_def=sess.graph_def)
        for step in xrange(input.FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = input.FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == input.FLAGS.max_steps:
                checkpoint_path = os.path.join(input.FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()
