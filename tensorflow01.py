import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1, matrix2)

# When a session is generated directly, the use of close
# is mandatory to free resources

# sess = tf.Session()
# result = sess.run(product)
# print result
# sess.close()

# Another approach is to use a WITH block

with tf.Session() as sess:
    result = sess.run([product])
    print result
