from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import input
import model

# Global constants
LABEL_PATH = input.LABEL_PATH
IMAGE_PATH = input.IMAGE_PATH
LABEL_SUFIX = input.LABEL_SUFIX
LABEL_FORMAT = input.LABEL_FORMAT
IMAGE_FORMAT = input.IMAGE_FORMAT
MOVING_AVERAGE_DECAY = input.MOVING_AVERAGE_DECAY


def evaluate_model(saver, summary_writer, top_k_op, summary_op):
	with tf.Session() as sess:
		
		# ckpt = tf.train.get_checkpoint_state(input.FLAGS.checkpoint_dir)
		ckpt = tf.train.latest_checkpoint(input.FLAGS.checkpoint_dir, latest_filename=None)

		#if ckpt and ckpt.model_checkpoint_path:
		if ckpt:
			print(ckpt)
			print(sess)
			#saver.restore(sess, ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt)
			#global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			global_step = ckpt.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return
		
	coord = tf.train.Coordinator()
	try:
		threads = []
		for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
			threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
		print(threads)
		num_iter = int(math.ceil(input.FLAGS.num_examples / input.FLAGS.batch_size))
		true_count = 0
		total_sample_count = num_iter * input.FLAGS.batch_size
		step = 0
		while step < num_iter and not coord.should_stop():
			predictions = sess.run([top_k_op])
			true_count += np.sum(predictions)
			step += 1

		# Compute precision @ 1.
		precision = true_count / total_sample_count
		print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
		summary = tf.Summary()
		summary.ParseFromString(sess.run(summary_op))
		summary.value.add(tag='Precision @ 1', simple_value=precision)
		summary_writer.add_summary(summary, global_step)
	except Exception as e:  # pylint: disable=broad-except
		coord.request_stop(e)

	coord.request_stop()
	coord.join(threads, stop_grace_period_secs=10)

def test():
	with tf.Graph().as_default():
		image, label = input.get_input(LABEL_PATH, LABEL_FORMAT, IMAGE_PATH, IMAGE_FORMAT)
		logits = model.inference(image)
		top_k_op = tf.nn.in_top_k(logits, label, 1)
		
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Get summaries for TENSOR BOARD
		summary_op = tf.merge_all_summaries()
		graph_def = tf.get_default_graph().as_graph_def()
		summary_writer = tf.train.SummaryWriter(input.FLAGS.eval_dir, graph_def=graph_def)

		while True:
			evaluate_model(saver, summary_writer, top_k_op, summary_op)
			if input.FLAGS.run_once:
				break
			time.sleep(input.FLAGS.eval_interval_secs)



def main(argv=None):
	test()

if __name__ == "__main__":
	tf.app.run()
