import tensorflow as tf
import os
from collections import namedtuple

from data.dataset_paths import TF_RECORDS_HOME

CropDef = namedtuple('CropDefinition', 'image_begin image_size crop_begin crop_size')
SizeDef = namedtuple('SizeDefinition', 'width height target_w target_h')
ShapeDef = namedtuple('ShapeDefinition', 'image_shape disparity_shape target_image_shape target_disparity_shape')


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecords(dataset):
    for subset in dataset:
        name = os.path.join(TF_RECORDS_HOME, '{}_{}.tfrecords'.format(dataset.name, subset.name))
        print('Creating tfrecords in {}'.format(name))
        writer = tf.python_io.TFRecordWriter(name)
        for example in subset.examples:
            tfexample = example.to_tfrecords()
            writer.write(tfexample.SerializeToString())
        writer.close()


def get_random_image_crop_tf(size_def, randomise=True):
    if randomise:
        begin_w = tf.random_uniform([], 0, size_def.width - size_def.target_w, dtype=tf.int32)
        begin_h = tf.random_uniform([], 0, size_def.height - size_def.target_h, dtype=tf.int32)
    else:
        begin_w = tf.constant(0, dtype=tf.int32, name='not_random_w')
        begin_h = tf.constant(50, dtype=tf.int32, name='not_random_h')

    image_begin = [begin_h, begin_w, 0]
    image_size = [size_def.target_h, size_def.target_w, -1]
    crop_begin = [begin_h // 4, begin_w // 4, 0]
    crop_size = [size_def.target_h // 4, size_def.target_w // 4, -1]
    return CropDef(image_begin, image_size, crop_begin, crop_size)


def read_and_decode(filename_queue, decoder):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    with tf.variable_scope('read_and_decode'):
        return decoder.decode(serialized_example)
