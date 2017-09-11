import tensorflow as tf

from models.image_sampler import reconstruct
from models.util import laplace, dssim


class Loss():
    def __init__(self, flags, config):
        self.sparse = config.get('sparse', True)
        self.max_disp = config.get('max_disp', flags.max_disp)
        self.losses = {}

    def build(self, output, placeholders):
        ground_truth = placeholders.d
        with tf.variable_scope('loss'):
            disparity_mask = tf.less_equal(ground_truth, self.max_disp)
            if self.sparse:
                sparse_mask = tf.greater(ground_truth, tf.constant(0.0))
                disparity_mask = tf.logical_and(disparity_mask, sparse_mask)
            disparity_mask = tf.cast(disparity_mask, tf.float32)
            N = tf.reduce_sum(disparity_mask)
            loss = tf.cond(N > tf.constant(0.0),
                           lambda: tf.reduce_sum(tf.abs(output - ground_truth) * disparity_mask) / N,
                           lambda: tf.constant(0.0))
            self.losses[output] = loss


class UnsupervisedLoss():
    def __init__(self, flags, config):
        self.smoothness_scale = config.get('smoothness_scale', 1e-2)
        self.losses = {}

    def build(self, output, placeholders):
        with tf.variable_scope('loss'):
            in_left_rec = reconstruct(placeholders.r, output, target_image='L')
            simmilarity_loss = tf.reduce_mean(tf.abs(placeholders.l - in_left_rec))
            smoothness_loss = self.smoothness_scale * tf.reduce_mean(tf.abs(laplace(output)))
            loss = simmilarity_loss + smoothness_loss
            self.losses[output] = loss


class UnsupervisedLossSSIM():
    def __init__(self, flags, config):
        self.smoothness_scale = config.get('smoothness_scale', 1e-2)
        self.ssim_ksize = config.get('ssim_ksize', 3)
        self.alpha = config.get('ssim_ksize', 0.85)
        self.losses = {}

    def build(self, output, placeholders):
        with tf.variable_scope('loss'):
            in_left_rec = reconstruct(placeholders.r, output, target_image='L')
            difference_abs = (1 - self.alpha) * tf.reduce_mean(tf.abs(placeholders.l - in_left_rec))
            difference_dssim = self.alpha * dssim(placeholders.l, in_left_rec, self.ssim_ksize)
            simmilarity_loss = difference_abs + difference_dssim
            smoothness_loss = self.smoothness_scale * tf.reduce_mean(tf.abs(laplace(output)))
            loss = simmilarity_loss + smoothness_loss
            self.losses[output] = loss
