import tensorflow as tf
import numpy as np


def conv_block(inputs, order, **kwargs):
    features = kwargs.get('features')
    ksize = kwargs.get('ksize')
    strides = kwargs.get('strides', (1, 1))
    padding = kwargs.get('padding', 'SAME')
    trainable = kwargs.get('trainable')
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    name = kwargs.get('name')
    initializer = tf.contrib.layers.xavier_initializer_conv2d() if not reuse else None
    with tf.variable_scope(name) as scope:
        if order == 'default':
            conv = tf.layers.conv2d(inputs, features, ksize, strides, padding, trainable=trainable,
                                    reuse=reuse, kernel_initializer=initializer, name='conv')
            return tf.contrib.layers.batch_norm(
                conv, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, trainable=trainable, fused=True,
                scope=scope
            )
        if order == 'dense':
            bn = tf.contrib.layers.batch_norm(
                inputs, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, trainable=trainable, fused=True,
                scope=scope
            )
            return tf.layers.conv2d(bn, features, ksize, strides, padding, trainable=trainable,
                                    reuse=reuse, kernel_initializer=initializer, name='conv')


def conv_bn_relu(inputs, **kwargs):
    return conv_block(inputs, order='default', **kwargs)


def bn_relu_conv(inputs, **kwargs):
    return conv_block(inputs, order='dense', **kwargs)


def conv_block_3d(inputs, order, **kwargs):
    features = kwargs.get('features')
    ksize = kwargs.get('ksize')
    strides = kwargs.get('strides', (1, 1, 1))
    padding = kwargs.get('padding', 'SAME')
    trainable = kwargs.get('trainable')
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    name = kwargs.get('name')
    initializer = tf.contrib.layers.xavier_initializer() if not reuse else None
    with tf.variable_scope(name) as scope:
        if order == 'default':
            conv = tf.layers.conv3d(inputs, features, ksize, strides, padding, trainable=trainable,
                                    reuse=reuse, kernel_initializer=initializer, name='conv')
            return tf.contrib.layers.batch_norm(
                conv, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, trainable=trainable, scope=scope
            )
        if order == 'dense':
            bn = tf.contrib.layers.batch_norm(
                inputs, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, trainable=trainable, scope=scope
            )
            return tf.layers.conv3d(bn, features, ksize, strides, padding, trainable=trainable,
                                    reuse=reuse, kernel_initializer=initializer, name='conv')


def conv_bn_relu_3d(inputs, **kwargs):
    return conv_block_3d(inputs, order='default', **kwargs)


def bn_relu_conv_3d(inputs, **kwargs):
    return conv_block_3d(inputs, order='dense', **kwargs)


def conv_3d_trans(inputs, **kwargs):
    features = kwargs.get('features')
    ksize = kwargs.get('ksize')
    strides = kwargs.get('strides', (1, 2, 2, 2, 1))
    padding = kwargs.get('padding', 'SAME')
    trainable = kwargs.get('trainable')
    reuse = kwargs.get('reuse')
    name = kwargs.get('name')
    out_shape = kwargs.get('out_shape')
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(name):
        with tf.variable_scope('conv') as conv_scope:
            if reuse:
                conv_scope.reuse_variables()
                kernel = tf.get_variable('kernel')
            else:
                kernel = tf.get_variable("kernel", (ksize, ksize, ksize, features, inputs.get_shape().as_list()[4]),
                                         initializer=initializer, trainable=trainable)
            return tf.nn.conv3d_transpose(inputs, kernel, output_shape=out_shape, strides=strides, padding=padding)


def conv_relu_bn_3d_trans(inputs, **kwargs):
    features = kwargs.get('features')
    ksize = kwargs.get('ksize')
    strides = kwargs.get('strides', (1, 2, 2, 2, 1))
    padding = kwargs.get('padding', 'SAME')
    trainable = kwargs.get('trainable')
    is_training = kwargs.get('is_training')
    reuse = kwargs.get('reuse')
    name = kwargs.get('name')
    out_shape = kwargs.get('out_shape')
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(name) as scope:
        with tf.variable_scope('conv') as conv_scope:
            if reuse:
                conv_scope.reuse_variables()
                kernel = tf.get_variable('kernel')
            else:
                kernel = tf.get_variable("kernel", (ksize, ksize, ksize, features, inputs.get_shape().as_list()[4]),
                                         initializer=initializer, trainable=trainable)
            conv = tf.nn.conv3d_transpose(inputs, kernel, output_shape=out_shape, strides=strides, padding=padding)
        return tf.contrib.layers.batch_norm(
            conv, activation_fn=tf.nn.relu, is_training=is_training, reuse=reuse, trainable=trainable, scope=scope
        )


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


def dssim(x, y, ksize):
    c1 = tf.constant(6.5, tf.float32)  # (k1 * L) ** 2
    c2 = tf.constant(58.5, tf.float32)  # (k2 * L) ** 2

    ksizes = (1, ksize, ksize, 1)

    mu_x = tf.nn.avg_pool(x, ksize=ksizes, strides=(1, 1, 1, 1), padding='VALID')
    mu_y = tf.nn.avg_pool(y, ksize=ksizes, strides=(1, 1, 1, 1), padding='VALID')

    sigma_x = tf.nn.avg_pool(x ** 2, ksize=ksizes, strides=(1, 1, 1, 1), padding='VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool(y ** 2, ksize=ksizes, strides=(1, 1, 1, 1), padding='VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(x * y, ksize=ksizes, strides=(1, 1, 1, 1), padding='VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    SSIM = SSIM_n / SSIM_d

    return tf.reduce_mean(tf.clip_by_value((1 - SSIM) / 2, 0, 1))
