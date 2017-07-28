import tensorflow as tf
from models.util import conv_bn_relu


class Resnet():
    def __init__(self, flags, config):
        self.num_res_blocks = config.get('num_res_blocks', 7)
        self.stem_features = config.get('stem_features', 32)
        self.stem_ksize = config.get('stem_ksize', 5)
        self.stem_strides = config.get('stem_strides', (2, 2))
        self.unary_features = config.get('unary_features', 32)
        self.unary_ksize = config.get('unary_ksize', 3)
        self.projection_features = config.get('projection_features', 32)
        self.projection_ksize = config.get('projection_ksize', 3)
        self.embeddings = {}

    def build(self, placeholder, is_training):
        with tf.variable_scope('unary'):
            left_image, right_image = placeholder.l, placeholder.r
            embedding_left = self.create_unary(left_image, is_training, None if is_training else True)
            embedding_right = self.create_unary(right_image, is_training, True)
            self.embeddings[placeholder] = (embedding_left, embedding_right)

    def create_unary(self, input, is_training, reuse):
        const_args = {
            'trainable': is_training,
            'is_training': is_training,
            'reuse': reuse
        }
        stem = conv_bn_relu(input, features=self.stem_features, ksize=self.stem_ksize, strides=self.stem_strides,
                            name='stemming', **const_args)
        layers = [stem]
        for i in range(self.num_res_blocks):
            with tf.variable_scope('resblock_{}'.format(i + 1)):
                conv_1 = conv_bn_relu(layers[-1], features=self.unary_features, ksize=self.unary_ksize,
                                      name='conv_1', **const_args)
                conv_2 = conv_bn_relu(conv_1, features=self.unary_features, ksize=self.unary_ksize,
                                      name='conv_2', **const_args)
                conv_3 = conv_bn_relu(conv_2, features=self.unary_features, ksize=self.unary_ksize,
                                      name='conv_3', **const_args)
                layers.append(tf.add(conv_1, conv_3))
        with tf.variable_scope('projection'):
            projection = tf.layers.conv2d(layers[-1], self.projection_features, self.projection_ksize, (1, 1), 'SAME',
                                          trainable=is_training,
                                          reuse=reuse, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv')
        return projection
