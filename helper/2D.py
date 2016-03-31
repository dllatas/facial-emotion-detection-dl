import tensorflow as tf
import numpy as np

"""
Let's initialize a couple of arrays that have a linear relationship
"""
# x_data is an array with 100 values created randomly
x_data = np.random.rand(100).astype("float32")
# y_data is an array that evaluates each value of x with the given expression
y_data = x_data * 0.1 + 0.3
# print x_data, y_data
"""
Now, from the given results, let's find the values for W and b, so:
y = x * W + b ... We already know these values, but the idea is that the
program can find them given the data supplied
"""
# The weight is defined by a random_uniform function between -1 and 1
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# bias is an array of zeros
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
# print W, b
"""
So far, only a memory definition has been allocated.
The type of memory is one of a tensorflow variable object.

On the next step, we will focus on the optimization problem.
The idea is to minimize the error between y and y_data
"""
# Loss stores the mean of the squared error between y and y_data
loss = tf.reduce_mean(tf.square(y - y_data))
# GDO is a factory function. It returns an Object initialized with a
# learning rate of 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# Implements the optimization defined on the previous line to the
# loss array
train = optimizer.minimize(loss)
"""
So far, that's all the logic that is needed to solve this problem.
The following steps are related to the use of tensorflow
"""

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(201):
    sess.run(train)
    if step % 10 == 0:
        print step, sess.run(W), sess.run(b)
