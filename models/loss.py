import tensorflow as tf


class Loss():
    def __init__(self, flags, config):
        self.sparse = config.get('sparse', True)
        self.max_disp = config.get('max_disp', flags.max_disp)
        self.losses = {}

    def build(self, output, ground_truth):
        with tf.variable_scope('loss'):
            disparity_mask = tf.less_equal(ground_truth, self.max_disp)
            if self.sparse:
                sparse_mask = tf.greater(ground_truth, tf.constant(0.0))
                disparity_mask = tf.cast(tf.logical_and(disparity_mask, sparse_mask), tf.float32)
            else:
                disparity_mask = tf.cast(disparity_mask, tf.float32)
            N = tf.reduce_sum(disparity_mask)
            loss = tf.cond(N > tf.constant(0.0),
                           lambda: tf.reduce_sum(tf.abs(output - ground_truth) * disparity_mask) / N,
                           lambda: tf.constant(0.0))
            self.losses[output] = loss
