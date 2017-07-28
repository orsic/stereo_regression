import tensorflow as tf
from collections import namedtuple

from data.tfrecords import get_random_image_crop_tf

CropDef = namedtuple('CropDefinition', 'image_begin image_size crop_begin crop_size')
SizeDef = namedtuple('SizeDefinition', 'width height target_w target_h')
ShapeDef = namedtuple('ShapeDefinition', 'image_shape disparity_shape target_image_shape target_disparity_shape')


class Decoder():
    tf_features = {
        'channels': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'name': tf.FixedLenFeature([], tf.string),
        'left': tf.FixedLenFeature([], tf.string),
        'right': tf.FixedLenFeature([], tf.string),
        'disparity': tf.FixedLenFeature([], tf.string),
    }

    def __init__(self, config):
        self.shapes = config.shapes
        self.config = config.flags
        self.is_training = config.is_training

    def decode(self, serialized_example):
        raise NotImplementedError()


class GenericDecoder(Decoder):
    def decode(self, serialized_example):
        with tf.variable_scope('decode'):
            features_to_decode = self.tf_features

            features = tf.parse_single_example(serialized_example, features=features_to_decode)

            width = tf.cast(features['width'], tf.int32)
            height = tf.cast(features['height'], tf.int32)
            channels = tf.constant(3, tf.int32)  # tf.cast(features['channels'], tf.int32)

            target_w = tf.constant(self.shapes.W, dtype=tf.int32) if self.shapes is not None else -1
            target_h = tf.constant(self.shapes.H, dtype=tf.int32) if self.shapes is not None else -1

            size_def = SizeDef(width, height, target_w, target_h)

            l = tf.decode_raw(features['left'], tf.float32)
            r = tf.decode_raw(features['right'], tf.float32)
            d = tf.decode_raw(features['disparity'], tf.float32)

            image_shape = tf.stack([size_def.height, size_def.width, channels])
            disparity_shape = tf.stack([size_def.height, size_def.width, 1])

            l = tf.reshape(l, image_shape)
            r = tf.reshape(r, image_shape)
            d = tf.reshape(d, disparity_shape)

        if self.shapes is not None:
            with tf.variable_scope('crop'):
                crop_def = get_random_image_crop_tf(size_def, self.is_training)
                l = tf.slice(l, crop_def.image_begin, crop_def.image_size, name='image_left')
                r = tf.slice(r, crop_def.image_begin, crop_def.image_size, name='image_right')
                d = tf.slice(d, crop_def.image_begin, crop_def.image_size, name='disparity')

        l, r, d = tf.train.batch([l, r, d], self.config.batch_size, self.config.num_threads, self.config.capacity,
                                 dynamic_pad=True)

        placeholders = dict()
        placeholders['l'] = l
        placeholders['r'] = r
        placeholders['d'] = d
        placeholders['name'] = features['name']

        return namedtuple('Placeholders', placeholders.keys())(**placeholders)


class KittiOdometryDecoder(Decoder):
    tf_features = {
        'channels': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'name': tf.FixedLenFeature([], tf.string),
        'left': tf.FixedLenFeature([], tf.string),
        'right': tf.FixedLenFeature([], tf.string),
    }

    def decode(self, serialized_example):
        features_to_decode = self.tf_features

        features = tf.parse_single_example(serialized_example, features=features_to_decode)

        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        channels = tf.cast(features['channels'], tf.int32)

        target_w = tf.constant(self.shapes.W, dtype=tf.int32)
        target_h = tf.constant(self.shapes.H, dtype=tf.int32)

        size_def = SizeDef(width, height, target_w, target_h)
        crop_def = get_random_image_crop_tf(size_def, self.config.randomise)

        l = tf.decode_raw(features['left'], tf.float32)
        r = tf.decode_raw(features['right'], tf.float32)

        image_shape = tf.stack([size_def.height, size_def.width, channels])

        l = tf.reshape(l, image_shape)
        r = tf.reshape(r, image_shape)

        l = tf.slice(l, crop_def.image_begin, crop_def.image_size, name='image_left')
        r = tf.slice(r, crop_def.image_begin, crop_def.image_size, name='image_right')

        l, r = tf.train.batch([l, r], self.config.batch_size, self.config.num_threads, self.config.capacity)

        placeholders = dict()
        placeholders['l'] = l
        placeholders['r'] = r
        placeholders['d'] = None
        placeholders['name'] = tf.decode_raw(features['name'], tf.string)

        return namedtuple('Placeholders', placeholders.keys())(**placeholders)


decoders = {
    'kitti': GenericDecoder,
    'kitti_submission': GenericDecoder,
    'sceneflow': GenericDecoder,
    'odometry': KittiOdometryDecoder,
}


def get_decoder_class(name):
    return decoders[name]
