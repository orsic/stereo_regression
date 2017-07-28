import tensorflow as tf

from models.util import conv_bn_relu_3d, conv_relu_bn_3d_trans, conv_3d_trans


class ResnetRegression():
    def __init__(self, flags, config):
        self.features = config.get('features', 32)
        self.ksize = config.get('ksize', 3)
        self.projections = {}

    def build(self, inputs, is_training, reuse):
        with tf.variable_scope('disparity_regression'):
            const_args = {
                'ksize': self.ksize,
                'trainable': is_training,
                'is_training': is_training,
                'reuse': reuse,
            }
            F = self.features
            with tf.variable_scope('resblock_1'):
                conv_1_1 = conv_bn_relu_3d(inputs, features=1 * F, name='conv_1', **const_args)
                conv_1_2 = conv_bn_relu_3d(conv_1_1, features=1 * F, name='conv_2', **const_args)
                res_1 = tf.add(conv_1_1, conv_1_2)
                conv_1_3 = conv_bn_relu_3d(res_1, features=2 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('resblock_2'):
                conv_2_1 = conv_bn_relu_3d(conv_1_3, features=2 * F, name='conv_1', **const_args)
                conv_2_2 = conv_bn_relu_3d(conv_2_1, features=2 * F, name='conv_2', **const_args)
                res_2 = tf.add(conv_2_1, conv_2_2)
                conv_2_3 = conv_bn_relu_3d(res_2, features=2 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('resblock_3'):
                conv_3_1 = conv_bn_relu_3d(conv_2_3, features=2 * F, name='conv_1', **const_args)
                conv_3_2 = conv_bn_relu_3d(conv_3_1, features=2 * F, name='conv_2', **const_args)
                res_3 = tf.add(conv_3_1, conv_3_2)
                conv_3_3 = conv_bn_relu_3d(res_3, features=2 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('resblock_4'):
                conv_4_1 = conv_bn_relu_3d(conv_3_3, features=2 * F, name='conv_1', **const_args)
                conv_4_2 = conv_bn_relu_3d(conv_4_1, features=2 * F, name='conv_2', **const_args)
                res_4 = tf.add(conv_4_1, conv_4_2)
                conv_4_3 = conv_bn_relu_3d(res_4, features=4 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('bridge'):
                b_1 = conv_bn_relu_3d(conv_4_3, features=4 * F, name='conv_1', **const_args)
                b_2 = conv_bn_relu_3d(b_1, features=4 * F, name='conv_2', **const_args)
            with tf.variable_scope('deconv_4'):
                os = tf.shape(conv_4_2)
                deconv_4 = conv_relu_bn_3d_trans(b_2, features=2 * F, name='deconv_4', **const_args, out_shape=os)
                ladder_4 = tf.add(conv_4_2, deconv_4)
            with tf.variable_scope('deconv_3'):
                os = tf.shape(conv_3_2)
                deconv_3 = conv_relu_bn_3d_trans(ladder_4, features=2 * F, name='deconv_3', **const_args, out_shape=os)
                ladder_3 = tf.add(conv_3_2, deconv_3)
            with tf.variable_scope('deconv_2'):
                os = tf.shape(conv_2_2)
                deconv_2 = conv_relu_bn_3d_trans(ladder_3, features=2 * F, name='deconv_2', **const_args, out_shape=os)
                ladder_2 = tf.add(conv_2_2, deconv_2)
            with tf.variable_scope('deconv_1'):
                os = tf.shape(conv_1_2)
                deconv_1 = conv_relu_bn_3d_trans(ladder_2, features=1 * F, name='deconv_1', **const_args, out_shape=os)
                ladder_1 = tf.add(conv_1_2, deconv_1)
            with tf.variable_scope('projection_3d'):
                ins = tf.shape(conv_1_2)
                N, D, W, H, C = ins[0], 2 * ins[1], 2 * ins[2], 2 * ins[3], 1
                os = (N, D, W, H, C)
                projection = conv_3d_trans(ladder_1, features=1, name='projection', **const_args, out_shape=os)
            self.projections[inputs] = projection

class ResnetRegressionConcat():
    def __init__(self, flags, config):
        self.features = config.get('features', 32)
        self.ksize = config.get('ksize', 3)
        self.projections = {}

    def build(self, inputs, is_training, reuse):
        with tf.variable_scope('disparity_regression'):
            const_args = {
                'ksize': self.ksize,
                'trainable': is_training,
                'is_training': is_training,
                'reuse': reuse,
            }
            F = self.features
            with tf.variable_scope('resblock_1'):
                conv_1_1 = conv_bn_relu_3d(inputs, features=F // 2, name='conv_1', **const_args)
                conv_1_2 = conv_bn_relu_3d(conv_1_1, features=F // 2, name='conv_2', **const_args)
                concat_1 = tf.concat((conv_1_1, conv_1_2), axis=-1)
                conv_1_3 = conv_bn_relu_3d(concat_1, features=2 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('resblock_2'):
                conv_2_1 = conv_bn_relu_3d(conv_1_3, features=2 * F, name='conv_1', **const_args)
                conv_2_2 = conv_bn_relu_3d(conv_2_1, features=2 * F, name='conv_2', **const_args)
                res_2 = tf.add(conv_2_1, conv_2_2)
                conv_2_3 = conv_bn_relu_3d(res_2, features=2 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('resblock_3'):
                conv_3_1 = conv_bn_relu_3d(conv_2_3, features=2 * F, name='conv_1', **const_args)
                conv_3_2 = conv_bn_relu_3d(conv_3_1, features=2 * F, name='conv_2', **const_args)
                res_3 = tf.add(conv_3_1, conv_3_2)
                conv_3_3 = conv_bn_relu_3d(res_3, features=2 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('resblock_4'):
                conv_4_1 = conv_bn_relu_3d(conv_3_3, features=2 * F, name='conv_1', **const_args)
                conv_4_2 = conv_bn_relu_3d(conv_4_1, features=2 * F, name='conv_2', **const_args)
                res_4 = tf.add(conv_4_1, conv_4_2)
                conv_4_3 = conv_bn_relu_3d(res_4, features=4 * F, name='conv_3', strides=(2, 2, 2), **const_args)
            with tf.variable_scope('bridge'):
                b_1 = conv_bn_relu_3d(conv_4_3, features=4 * F, name='conv_1', **const_args)
                b_2 = conv_bn_relu_3d(b_1, features=4 * F, name='conv_2', **const_args)
            with tf.variable_scope('deconv_4'):
                os = tf.shape(conv_4_2)
                deconv_4 = conv_relu_bn_3d_trans(b_2, features=2 * F, name='deconv_4', **const_args, out_shape=os)
                ladder_4 = tf.add(conv_4_2, deconv_4)
            with tf.variable_scope('deconv_3'):
                os = tf.shape(conv_3_2)
                deconv_3 = conv_relu_bn_3d_trans(ladder_4, features=2 * F, name='deconv_3', **const_args, out_shape=os)
                ladder_3 = tf.add(conv_3_2, deconv_3)
            with tf.variable_scope('deconv_2'):
                os = tf.shape(conv_2_2)
                deconv_2 = conv_relu_bn_3d_trans(ladder_3, features=2 * F, name='deconv_2', **const_args, out_shape=os)
                ladder_2 = tf.add(conv_2_2, deconv_2)
            with tf.variable_scope('deconv_1'):
                os = tf.shape(conv_1_2)
                deconv_1 = conv_relu_bn_3d_trans(ladder_2, features=F // 2, name='deconv_1', **const_args, out_shape=os)
                ladder_1 = tf.concat((conv_1_2, deconv_1), axis=-1)
            with tf.variable_scope('projection_3d'):
                ins = tf.shape(conv_1_2)
                N, D, W, H, C = ins[0], 2 * ins[1], 2 * ins[2], 2 * ins[3], 1
                os = (N, D, W, H, C)
                projection = conv_3d_trans(ladder_1, features=1, name='projection', **const_args, out_shape=os)
            self.projections[inputs] = projection